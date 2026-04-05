"""
AVTrack ONNX Export (PC)
=========================
Converts the AVTrack-DeiT model to ONNX.
AM is disabled while preserving view-invariant learning.

Usage:
    cd <AVTrack_ROOT>
    python export_avtrack_onnx.py \
        --checkpoint model.pth \
        --output avtrack.onnx
    python avtrack_onnx_exporter.py \ 
        --checkpoint ~/pipeline/models/AVTrack_model.pth \ 
        --output ~/pipeline/models/avtrack.onnx

    Then test:
    python export_avtrack_onnx.py \
        --checkpoint model.pth \
        --output avtrack.onnx \
        --test-video test_video.mp4
"""

import os, sys, types, argparse, time
import numpy as np
import torch
import torch.nn as nn

sys.path.append(os.getcwd())


# =============================================================================
# 1) Load model + remove AM
# =============================================================================

def load_model(checkpoint_path, config_name):
    from lib.config.avtrack.config import cfg, update_config_from_file
    from lib.models.avtrack import build_avtrack
    from lib.models.avtrack.utils import combine_tokens, recover_tokens

    yaml_path = os.path.join('experiments', 'avtrack', f'{config_name}.yaml')
    if os.path.exists(yaml_path):
        update_config_from_file(yaml_path)
        print(f"[OK] Config: {yaml_path}")

    model = build_avtrack(cfg, training=False)

    ckpt = torch.load(checkpoint_path, map_location='cpu')
    sd = ckpt.get('net', ckpt.get('state_dict', ckpt.get('model', ckpt))) if isinstance(ckpt, dict) else ckpt
    clean = {k.replace('module.', ''): v for k, v in sd.items()}
    model.load_state_dict(clean, strict=False)
    print(f"[OK] Checkpoint: {checkpoint_path}")

    # Remove AM
    def fwd_no_am(self, z, x, is_distill=False):
        x = self.patch_embed(x)
        z = self.patch_embed(z)
        z += self.pos_embed_z
        x += self.pos_embed_x
        x = combine_tokens(z, x)
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        lens_z = self.pos_embed_z.shape[1]
        lens_x = self.pos_embed_x.shape[1]
        x = recover_tokens(x, lens_z, lens_x)
        return self.norm(x), {"attn": None, "probs_active": []}

    def fwd_backbone(self, z, x, **kwargs):
        return self.forward_features(z, x)

    model.backbone.forward_features = types.MethodType(fwd_no_am, model.backbone)
    model.backbone.forward = types.MethodType(fwd_backbone, model.backbone)
    print("[OK] AM disabled")

    return model, cfg


# =============================================================================
# 2) ONNX export wrapper
# =============================================================================

class AVTrackONNX(nn.Module):
    """
    Wrapper for ONNX export.
    Input:  template [1,3,128,128], search [1,3,S,S]
    Output: score_map, size_map, offset_map, pred_boxes
    """
    def __init__(self, model):
        super().__init__()
        self.backbone = model.backbone
        self.box_head = model.box_head
        self.feat_len_s = model.feat_len_s
        self.feat_sz_s = model.feat_sz_s

    def forward(self, template, search):
        x, _ = self.backbone(z=template, x=search)
        feat = x[-1] if isinstance(x, list) else x
        enc_opt = feat[:, -self.feat_len_s:]
        opt = enc_opt.unsqueeze(-1).permute(0, 3, 2, 1).contiguous()
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)
        score_map, pred_boxes, size_map, offset_map = self.box_head(opt_feat, None)
        return score_map, pred_boxes, size_map, offset_map


# =============================================================================
# 3) Export
# =============================================================================

def export_onnx(model, cfg, output_path, opset=17):
    wrapper = AVTrackONNX(model)
    wrapper.eval()

    t_size = cfg.TEST.TEMPLATE_SIZE  # 128
    s_size = cfg.TEST.SEARCH_SIZE    # 320
    print(f"[INFO] Template: {t_size}, Search: {s_size}")

    t_dummy = torch.randn(1, 3, t_size, t_size)
    s_dummy = torch.randn(1, 3, s_size, s_size)

    # Test forward
    with torch.no_grad():
        out = wrapper(t_dummy, s_dummy)
        print(f"[OK] Forward OK | score:{out[0].shape} size:{out[1].shape} offset:{out[2].shape}")

    # Export
    print(f"[INFO] ONNX export -> {output_path} (opset {opset})")
    torch.onnx.export(
        wrapper, (t_dummy, s_dummy), output_path,
        dynamic_axes={
        'search': {2: 'h', 3: 'w'},
        'score_map': {2: 'h', 3: 'w'},
        'pred_boxes': {2: 'h', 3: 'w'}
    },
        export_params=True,
        opset_version=opset,
        do_constant_folding=False,
        input_names=['template', 'search'],
        output_names=['score_map', 'pred_boxes', 'size_map', 'offset_map'],
    )
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"[OK] Export completed: {output_path} ({size_mb:.1f} MB)")

    # Simplify
    try:
        import onnx
        from onnxsim import simplify
        m = onnx.load(output_path)
        onnx.checker.check_model(m)
        m_simp, ok = simplify(m)
        if ok:
            onnx.save(m_simp, output_path)
            new_size = os.path.getsize(output_path) / 1024 / 1024
            print(f"[OK] Simplified: {new_size:.1f} MB")
    except ImportError:
        print("[WARNING] onnxsim not found, run: pip install onnxsim")
    except Exception as e:
        print(f"[WARNING] Simplification failed: {e}")

    return t_size, s_size


# =============================================================================
# 4) Verification
# =============================================================================

def verify_onnx(output_path, model, cfg):
    """Compares PyTorch outputs with ONNX Runtime outputs."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("[WARNING] onnxruntime not found, run: pip install onnxruntime-gpu")
        return

    t_size = cfg.TEST.TEMPLATE_SIZE
    s_size = cfg.TEST.SEARCH_SIZE

    t_dummy = torch.randn(1, 3, t_size, t_size)
    s_dummy = torch.randn(1, 3, s_size, s_size)

    # PyTorch
    wrapper = AVTrackONNX(model).eval()
    with torch.no_grad():
        pt_out = [o.numpy() for o in wrapper(t_dummy, s_dummy)]

    # ONNX Runtime
    sess = ort.InferenceSession(output_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    ort_out = sess.run(None, {'template': t_dummy.numpy(), 'search': s_dummy.numpy()})

    names = ['score_map', 'pred_boxes', 'size_map', 'offset_map']
    print("\n  Verification:")
    all_ok = True
    for name, pt, ort_o in zip(names, pt_out, ort_out):
        diff = np.max(np.abs(pt - ort_o))
        status = "OK" if diff < 1e-4 else "WARNING" if diff < 1e-2 else "ERROR"
        if status != "OK": all_ok = False
        print(f"    {name:12s} max_diff={diff:.6f} [{status}]")
    print(f"  {'SUCCESS' if all_ok else 'DIFFERENCE DETECTED - please review'}")

    # FPS benchmark
    print("\n  ORT FPS Benchmark (100 frame)...")
    for _ in range(10):  # warmup
        sess.run(None, {'template': t_dummy.numpy(), 'search': s_dummy.numpy()})
    t0 = time.time()
    for _ in range(100):
        sess.run(None, {'template': t_dummy.numpy(), 'search': s_dummy.numpy()})
    elapsed = time.time() - t0
    print(f"    ONNX Runtime: {100/elapsed:.0f} FPS (3060 Laptop)")


# =============================================================================
# 5) Video test (with ONNX Runtime)
# =============================================================================

def test_video_ort(onnx_path, video_src, cfg):
    """Video tracking test with ONNX Runtime."""
    import onnxruntime as ort
    import cv2

    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    t_size = cfg.TEST.TEMPLATE_SIZE
    s_size = cfg.TEST.SEARCH_SIZE
    t_factor = cfg.TEST.TEMPLATE_FACTOR
    s_factor = cfg.TEST.SEARCH_FACTOR

    def preprocess(image, size):
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
        img = (img - MEAN) / STD
        return img.transpose(2, 0, 1)[np.newaxis].astype(np.float32)

    def crop(frame, cx, cy, wh_area, out_size, factor):
        crop_sz = max(int(np.ceil(np.sqrt(wh_area) * factor)), 1)
        x1, y1 = int(cx - crop_sz/2), int(cy - crop_sz/2)
        x2, y2 = x1 + crop_sz, y1 + crop_sz
        H, W = frame.shape[:2]
        pads = [max(0,-y1), max(0,y2-H), max(0,-x1), max(0,x2-W)]
        x1c, y1c, x2c, y2c = max(0,x1), max(0,y1), min(W,x2), min(H,y2)
        c = frame[y1c:y2c, x1c:x2c]
        if any(p>0 for p in pads):
            avg = np.mean(c, axis=(0,1)).astype(np.uint8).tolist()
            c = cv2.copyMakeBorder(c, *pads, cv2.BORDER_CONSTANT, value=avg)
        rf = out_size / crop_sz
        return cv2.resize(c, (out_size, out_size), interpolation=cv2.INTER_LINEAR), rf

    sess = ort.InferenceSession(onnx_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    cap = cv2.VideoCapture(video_src)
    ret, frame = cap.read()
    if not ret: return

    bbox = cv2.selectROI("ONNX Test", frame, fromCenter=False)
    cv2.destroyWindow("ONNX Test")
    x, y, w, h = bbox
    cx, cy = x + w/2, y + h/2

    tmpl, _ = crop(frame, cx, cy, w*h, t_size, t_factor)
    tmpl_np = preprocess(tmpl, t_size)

    while True:
        ret, frame = cap.read()
        if not ret: break
        t0 = time.time()

        srch, rf = crop(frame, cx, cy, w*h, s_size, s_factor)
        srch_np = preprocess(srch, s_size)

        score_map, size_map, offset_map = sess.run(
            None, {'template': tmpl_np, 'search': srch_np})

        # Decode: find the max location from score_map instead of pred_boxes
        sm = score_map[0, 0]
        my, mx = np.unravel_index(np.argmax(sm), sm.shape)
        pw, ph = size_map[0, 0, my, mx], size_map[0, 1, my, mx]
        ox, oy = offset_map[0, 0, my, mx], offset_map[0, 1, my, mx]
        stride = s_size / sm.shape[0]

        cx += ((mx + ox) * stride - s_size/2) / rf
        cy += ((my + oy) * stride - s_size/2) / rf
        w = pw * s_size / rf
        h = ph * s_size / rf
        score = float(sm.max())

        fps = 1.0 / (time.time() - t0 + 1e-9)
        x1, y1 = int(cx-w/2), int(cy-h/2)
        cv2.rectangle(frame, (x1,y1), (x1+int(w),y1+int(h)), (0,255,0), 2)
        cv2.putText(frame, f"ORT FPS:{fps:.0f} score:{score:.2f}",
                    (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        cv2.imshow("ONNX Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--config', type=str, default='deit_tiny_patch16_224')
    parser.add_argument('--output', type=str, default='avtrack.onnx')
    parser.add_argument('--opset', type=int, default=16)
    parser.add_argument('--test-video', type=str, default=None,
                        help='If provided, runs a video test using ONNX Runtime')
    args = parser.parse_args()

    model, cfg = load_model(args.checkpoint, args.config)
    t_size, s_size = export_onnx(model, cfg, args.output, args.opset)
    verify_onnx(args.output, model, cfg)

    if args.test_video:
        src = int(args.test_video) if args.test_video.isdigit() else args.test_video
        test_video_ort(args.output, src, cfg)

    print(f"\n  Next step (Jetson Orin Nano):")
    print(f"    trtexec --onnx={args.output} --saveEngine=avtrack_fp16.engine --fp16")

if __name__ == '__main__':
    main()