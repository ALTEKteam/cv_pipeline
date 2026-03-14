"""
MixFormerV2 ONNX Export (PC)
==============================
Converts the MixFormerV2 model to ONNX.

Usage:
    cd <MixFormerV2_ROOT>
    python converter/mixformerv2_onnx_exporter.py \
        --checkpoint model.pth \
        --output mixformerv2.onnx
    python converter/mixformerv2_onnx_exporter.py \
        --checkpoint /path/to/MixFormerV2_model.pth \
        --output /path/to/output/mixformerv2.onnx

    Then test:
    python converter/mixformerv2_onnx_exporter.py \
        --checkpoint model.pth \
        --output mixformerv2.onnx \
        --test-video test_video.mp4
"""

import os, sys, argparse, time
import numpy as np
import torch
import torch.nn as nn

sys.path.append(os.getcwd())


# =============================================================================
# 1) Load model
# =============================================================================

def load_model(checkpoint_path, config_name):
    from lib.config.mixformer2_vit.config import cfg, update_config_from_file
    from lib.models.mixformer2_vit.mixformer2_vit import build_mixformer_vit

    yaml_path = os.path.join('experiments', 'mixformer2_vit', f'{config_name}.yaml')
    if os.path.exists(yaml_path):
        update_config_from_file(yaml_path)
        print(f"[OK] Config: {yaml_path}")

    model = build_mixformer_vit(cfg, train=False)

    ckpt = torch.load(checkpoint_path, map_location='cpu')
    sd = ckpt.get('net', ckpt.get('state_dict', ckpt.get('model', ckpt))) if isinstance(ckpt, dict) else ckpt
    clean = {k.replace('module.', ''): v for k, v in sd.items()}
    model.load_state_dict(clean, strict=False)
    print(f"[OK] Checkpoint: {checkpoint_path}")

    return model, cfg


# =============================================================================
# 2) ONNX export wrapper
# =============================================================================

class MixFormerV2ONNX(nn.Module):
    """
    Wrapper for ONNX export.
    Input:  template [1,3,T,T], online_template [1,3,T,T], search [1,3,S,S]
    Output: pred_boxes [1,1,4] (cxcywh normalized)
    """
    def __init__(self, model, feat_sz, img_sz):
        super().__init__()
        self.backbone = model.backbone
        self.box_head = model.box_head
        self.feat_sz = feat_sz
        self.img_sz = img_sz

    def forward(self, template, online_template, search):
        _, _, _, reg_tokens, _ = self.backbone(template, online_template, search)

        # MLP head forward: process 4 regression tokens (L, R, T, B)
        reg_token_l, reg_token_r, reg_token_t, reg_token_b = reg_tokens.unbind(dim=1)

        score_l = self.box_head.layers(reg_token_l)
        score_r = self.box_head.layers(reg_token_r)
        score_t = self.box_head.layers(reg_token_t)
        score_b = self.box_head.layers(reg_token_b)

        prob_l = score_l.softmax(dim=-1)
        prob_r = score_r.softmax(dim=-1)
        prob_t = score_t.softmax(dim=-1)
        prob_b = score_b.softmax(dim=-1)

        # Compute coordinates using indices (avoid CUDA-bound self.indice)
        stride = self.img_sz / self.feat_sz
        indice = torch.arange(0, self.feat_sz, device=reg_tokens.device,
                              dtype=reg_tokens.dtype).unsqueeze(0) * stride

        coord_l = torch.sum(indice * prob_l, dim=-1)
        coord_r = torch.sum(indice * prob_r, dim=-1)
        coord_t = torch.sum(indice * prob_t, dim=-1)
        coord_b = torch.sum(indice * prob_b, dim=-1)

        # xyxy -> cxcywh (normalized)
        pred_xyxy = torch.stack((coord_l, coord_t, coord_r, coord_b), dim=1) / self.img_sz
        cx = (pred_xyxy[:, 0] + pred_xyxy[:, 2]) / 2
        cy = (pred_xyxy[:, 1] + pred_xyxy[:, 3]) / 2
        w = pred_xyxy[:, 2] - pred_xyxy[:, 0]
        h = pred_xyxy[:, 3] - pred_xyxy[:, 1]
        pred_boxes = torch.stack((cx, cy, w, h), dim=1).unsqueeze(1)

        return pred_boxes, prob_l, prob_t, prob_r, prob_b


# =============================================================================
# 3) Export
# =============================================================================

def export_onnx(model, cfg, output_path, opset=17):
    feat_sz = cfg.MODEL.FEAT_SZ
    s_size = cfg.DATA.SEARCH.SIZE
    wrapper = MixFormerV2ONNX(model, feat_sz, s_size)
    wrapper.eval()

    t_size = cfg.DATA.TEMPLATE.SIZE
    print(f"[INFO] Template: {t_size}, Search: {s_size}, Feat: {feat_sz}")

    t_dummy = torch.randn(1, 3, t_size, t_size)
    ot_dummy = torch.randn(1, 3, t_size, t_size)
    s_dummy = torch.randn(1, 3, s_size, s_size)

    # Test forward
    with torch.no_grad():
        out = wrapper(t_dummy, ot_dummy, s_dummy)
        print(f"[OK] Forward OK | pred_boxes:{out[0].shape} "
              f"prob_l:{out[1].shape} prob_t:{out[2].shape} "
              f"prob_r:{out[3].shape} prob_b:{out[4].shape}")

    # Export
    print(f"[INFO] ONNX export -> {output_path} (opset {opset})")
    torch.onnx.export(
        wrapper, (t_dummy, ot_dummy, s_dummy), output_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=['template', 'online_template', 'search'],
        output_names=['pred_boxes', 'prob_l', 'prob_t', 'prob_r', 'prob_b'],
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

    feat_sz = cfg.MODEL.FEAT_SZ
    t_size = cfg.DATA.TEMPLATE.SIZE
    s_size = cfg.DATA.SEARCH.SIZE

    t_dummy = torch.randn(1, 3, t_size, t_size)
    ot_dummy = torch.randn(1, 3, t_size, t_size)
    s_dummy = torch.randn(1, 3, s_size, s_size)

    # PyTorch
    wrapper = MixFormerV2ONNX(model, feat_sz, s_size).eval()
    with torch.no_grad():
        pt_out = [o.numpy() for o in wrapper(t_dummy, ot_dummy, s_dummy)]

    # ONNX Runtime
    sess = ort.InferenceSession(output_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    ort_out = sess.run(None, {
        'template': t_dummy.numpy(),
        'online_template': ot_dummy.numpy(),
        'search': s_dummy.numpy()
    })

    names = ['pred_boxes', 'prob_l', 'prob_t', 'prob_r', 'prob_b']
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
    feed = {
        'template': t_dummy.numpy(),
        'online_template': ot_dummy.numpy(),
        'search': s_dummy.numpy()
    }
    for _ in range(10):  # warmup
        sess.run(None, feed)
    t0 = time.time()
    for _ in range(100):
        sess.run(None, feed)
    elapsed = time.time() - t0
    print(f"    ONNX Runtime: {100/elapsed:.0f} FPS")


# =============================================================================
# 5) Video test (with ONNX Runtime)
# =============================================================================

def test_video_ort(onnx_path, video_src, cfg):
    """Video tracking test with ONNX Runtime."""
    import onnxruntime as ort
    import cv2

    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    t_size = cfg.DATA.TEMPLATE.SIZE
    s_size = cfg.DATA.SEARCH.SIZE
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
    # Use initial template as online_template as well
    ot_np = tmpl_np.copy()

    while True:
        ret, frame = cap.read()
        if not ret: break
        t0 = time.time()

        srch, rf = crop(frame, cx, cy, w*h, s_size, s_factor)
        srch_np = preprocess(srch, s_size)

        pred_boxes, *_ = sess.run(
            None, {'template': tmpl_np, 'online_template': ot_np, 'search': srch_np})

        # Decode pred_boxes (cxcywh normalized)
        pcx, pcy, pw, ph = pred_boxes[0, 0]
        # Convert from normalized to search image coordinates
        pred_cx = pcx * s_size
        pred_cy = pcy * s_size
        pred_w = pw * s_size
        pred_h = ph * s_size

        # Map back to original frame coordinates
        cx += (pred_cx - s_size/2) / rf
        cy += (pred_cy - s_size/2) / rf
        w = pred_w / rf
        h = pred_h / rf

        fps = 1.0 / (time.time() - t0 + 1e-9)
        x1, y1 = int(cx-w/2), int(cy-h/2)
        cv2.rectangle(frame, (x1,y1), (x1+int(w),y1+int(h)), (0,255,0), 2)
        cv2.putText(frame, f"ORT FPS:{fps:.0f}",
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
    parser.add_argument('--config', type=str, default='teacher_288_depth12')
    parser.add_argument('--output', type=str, default='mixformerv2.onnx')
    parser.add_argument('--opset', type=int, default=17)
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
    print(f"    trtexec --onnx={args.output} --saveEngine=mixformerv2_fp16.engine --fp16")

if __name__ == '__main__':
    main()
