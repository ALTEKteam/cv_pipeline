"""
AVTrack Pipeline Adapter (without AM head)
===========================================
Supports 3 backends: TensorRT > ONNX Runtime > PyTorch
All postprocessing improvements included:
  - Hanning window center bias
  - Local weighted argmax (sub-pixel refinement)
  - EMA smoothing for position and size
  - Frame-to-frame + absolute bbox clipping
  - Periodic template update
"""

import os
import sys
import types
import numpy as np
import cv2 as cv
from config import AVTRACK_ENGINE_PATH, AVTRACK_ROOT
import torch
import onnxruntime as ort

sys.path.append(AVTRACK_ROOT)


# =============================================================================
# Load AVTrack model and configuration
# =============================================================================

def _load_avtrack(config_name, checkpoint_path):
    from lib.config.avtrack.config import cfg, update_config_from_file
    from lib.models.avtrack import build_avtrack
    from lib.models.avtrack.utils import combine_tokens, recover_tokens

    yaml_path = os.path.join(os.path.dirname(__file__), '..',
                             'experiments', 'avtrack', f'{config_name}.yaml')
    if not os.path.exists(yaml_path):
        for base in sys.path:
            candidate = os.path.join(base, 'experiments', 'avtrack', f'{config_name}.yaml')
            if os.path.exists(candidate):
                yaml_path = candidate
                break

    if os.path.exists(yaml_path):
        update_config_from_file(yaml_path)

    model = build_avtrack(cfg, training=False)

    ckpt = torch.load(checkpoint_path, map_location='cpu')
    sd = ckpt.get('net', ckpt.get('state_dict', ckpt.get('model', ckpt))) if isinstance(ckpt, dict) else ckpt
    clean = {k.replace('module.', ''): v for k, v in sd.items()}
    model.load_state_dict(clean, strict=False)

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
        return self.forward_features(z, x, kwargs.get('is_distill', False))

    model.backbone.forward_features = types.MethodType(fwd_no_am, model.backbone)
    model.backbone.forward = types.MethodType(fwd_backbone, model.backbone)
    model.eval()
    return model, cfg


# =============================================================================
# Pre/post processing
# =============================================================================

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _preprocess(image, size):
    img = cv.cvtColor(image, cv.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = cv.resize(img, (size, size), interpolation=cv.INTER_LINEAR)
    img = (img - MEAN) / STD
    return torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float().pin_memory()


def _preprocess_np(image, size):
    """NumPy version for ORT/TRT — avoids torch overhead."""
    img = cv.cvtColor(image, cv.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = cv.resize(img, (size, size), interpolation=cv.INTER_LINEAR)
    img = (img - MEAN) / STD
    return img.transpose(2, 0, 1)[np.newaxis].astype(np.float32)


def _crop(frame, cx, cy, wh_area, out_size, factor):
    crop_sz = max(int(np.ceil(np.sqrt(wh_area) * factor)), 1)
    x1, y1 = int(cx - crop_sz / 2), int(cy - crop_sz / 2)
    x2, y2 = x1 + crop_sz, y1 + crop_sz
    H, W = frame.shape[:2]
    pads = [max(0, -y1), max(0, y2 - H), max(0, -x1), max(0, x2 - W)]
    x1c, y1c = max(0, x1), max(0, y1)
    x2c, y2c = min(W, x2), min(H, y2)
    cropped = frame[y1c:y2c, x1c:x2c]
    if any(p > 0 for p in pads):
        avg = np.mean(cropped, axis=(0, 1)).astype(np.uint8).tolist()
        cropped = cv.copyMakeBorder(cropped, *pads, cv.BORDER_CONSTANT, value=avg)
    rf = out_size / crop_sz
    return cv.resize(cropped, (out_size, out_size), interpolation=cv.INTER_LINEAR), rf, crop_sz


# =============================================================================
# Shared decode logic
# =============================================================================

def _decode_score_map(sm, hanning, search_size, size_map, offset_map):
    """
    Decode score_map with hanning window + local weighted argmax.
    Returns: mx, my (float, sub-pixel), max_score, pw, ph, ox, oy
    """
    sm_w = sm * 0.65 + hanning * 0.35

    raw_my, raw_mx = np.unravel_index(np.argmax(sm_w), sm_w.shape)
    max_score = float(sm[raw_my, raw_mx])

    # Local 3x3 weighted average for sub-pixel refinement
    h, w = sm.shape
    y_lo, y_hi = max(0, raw_my - 1), min(h, raw_my + 2)
    x_lo, x_hi = max(0, raw_mx - 1), min(w, raw_mx + 2)

    patch = sm[y_lo:y_hi, x_lo:x_hi]
    patch = np.maximum(patch, 0)
    patch_sum = patch.sum()

    if patch_sum > 1e-6:
        ys, xs = np.meshgrid(np.arange(y_lo, y_hi), np.arange(x_lo, x_hi), indexing='ij')
        mx = float((patch * xs).sum() / patch_sum)
        my = float((patch * ys).sum() / patch_sum)
    else:
        mx, my = float(raw_mx), float(raw_my)

    pw = size_map[0, 0, raw_my, raw_mx]
    ph = size_map[0, 1, raw_my, raw_mx]
    ox = offset_map[0, 0, raw_my, raw_mx]
    oy = offset_map[0, 1, raw_my, raw_mx]

    return mx, my, max_score, pw, ph, ox, oy


def _update_state(tracker, mx, my, ox, oy, pw, ph, rf):
    """Update tracker cx/cy/w/h with EMA smoothing and bbox clipping."""
    stride = tracker.search_size / tracker._sm_size
    alpha = 0.6

    new_cx = tracker.cx + ((mx + ox) * stride - tracker.search_size / 2) / rf
    new_cy = tracker.cy + ((my + oy) * stride - tracker.search_size / 2) / rf
    new_w = pw * tracker.search_size / rf
    new_h = ph * tracker.search_size / rf

    # Frame-to-frame: max 20% change per frame
    new_w = np.clip(new_w, tracker.w * 0.8, tracker.w * 1.2)
    new_h = np.clip(new_h, tracker.h * 0.8, tracker.h * 1.2)

    # Absolute: max 3x init size
    new_w = np.clip(new_w, tracker._init_w * 0.2, tracker._init_w * 3)
    new_h = np.clip(new_h, tracker._init_h * 0.2, tracker._init_h * 3)

    # EMA smoothing
    tracker.cx = alpha * new_cx + (1 - alpha) * tracker.cx
    tracker.cy = alpha * new_cy + (1 - alpha) * tracker.cy
    tracker.w = alpha * new_w + (1 - alpha) * tracker.w
    tracker.h = alpha * new_h + (1 - alpha) * tracker.h


# =============================================================================
# AVTrack Tracker
# =============================================================================

class AVTrackTracker:
    def __init__(self, config_name='deit_tiny_patch16_224', checkpoint_path=None,
                 onnx_path=None, engine_path=None, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.use_trt = engine_path is not None
        self.use_ort = onnx_path is not None and not self.use_trt

        # --- TensorRT ---
        if self.use_trt:
            import tensorrt as trt

            self.trt_logger = trt.Logger(trt.Logger.WARNING)
            with open(engine_path, 'rb') as f:
                runtime = trt.Runtime(self.trt_logger)
                self.trt_engine = runtime.deserialize_cuda_engine(f.read())
            self.trt_context = self.trt_engine.create_execution_context()
            self.cuda_stream = torch.cuda.Stream()

            self.trt_inputs = {}
            self.trt_outputs = {}
            for i in range(self.trt_engine.num_io_tensors):
                name = self.trt_engine.get_tensor_name(i)
                shape = tuple(self.trt_engine.get_tensor_shape(name))
                tensor = torch.empty(shape, dtype=torch.float32, device='cuda')
                self.trt_context.set_tensor_address(name, tensor.data_ptr())
                mode = self.trt_engine.get_tensor_mode(name)
                if mode == trt.TensorIOMode.INPUT:
                    self.trt_inputs[name] = tensor
                else:
                    self.trt_outputs[name] = tensor

            from lib.config.avtrack.config import cfg, update_config_from_file
            for base in sys.path:
                yp = os.path.join(base, 'experiments', 'avtrack', f'{config_name}.yaml')
                if os.path.exists(yp):
                    update_config_from_file(yp)
                    break
            self.cfg = cfg
            self.model = None
            self.sess = None

            # TRT warmup
            for _ in range(5):
                with torch.cuda.stream(self.cuda_stream):
                    self.trt_context.execute_async_v3(stream_handle=self.cuda_stream.cuda_stream)
                self.cuda_stream.synchronize()
            print(f"[AVTrack] TensorRT mode | {engine_path}")

        # --- ONNX Runtime ---
        elif self.use_ort:
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.enable_mem_pattern = True
            sess_options.enable_cpu_mem_arena = True

            self.sess = ort.InferenceSession(
                onnx_path,
                sess_options=sess_options,
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            from lib.config.avtrack.config import cfg, update_config_from_file
            yaml_path = None
            for base in sys.path:
                candidate = os.path.join(base, 'experiments', 'avtrack', f'{config_name}.yaml')
                if os.path.exists(candidate):
                    yaml_path = candidate
                    break
            if yaml_path:
                update_config_from_file(yaml_path)
            self.cfg = cfg
            self.model = None
            print(f"[AVTrack] ONNX Runtime mode | {onnx_path}")

        # --- PyTorch ---
        else:
            if checkpoint_path is None:
                raise ValueError("checkpoint_path, onnx_path, or engine_path required")
            self.model, self.cfg = _load_avtrack(config_name, checkpoint_path)
            self.model.to(self.device)
            self.sess = None
            print(f"[AVTrack] PyTorch mode | {self.device}")

        # Common config
        self.template_size = self.cfg.TEST.TEMPLATE_SIZE
        self.search_size = self.cfg.TEST.SEARCH_SIZE
        self.template_factor = self.cfg.TEST.TEMPLATE_FACTOR
        self.search_factor = self.cfg.TEST.SEARCH_FACTOR

        self.template_tensor = None
        self.template_np = None
        self.cx = self.cy = self.w = self.h = 0
        self._init_w = self._init_h = 0
        self._frame_count = 0
        self._hanning = None
        self._sm_size = 0

        # ORT warmup
        if self.use_ort and self.sess is not None:
            dummy_t = np.zeros((1, 3, self.template_size, self.template_size), dtype=np.float32)
            dummy_s = np.zeros((1, 3, self.search_size, self.search_size), dtype=np.float32)
            for _ in range(5):
                self.sess.run(None, {'template': dummy_t, 'search': dummy_s})
            print("[AVTrack] ORT warmup done")

    # =========================================================================
    # Initialize
    # =========================================================================

    def initialize(self, frame, info):
        """Start tracking with bbox from YOLO. info = {'init_bbox': [x,y,w,h]}"""
        bbox = info['init_bbox'] if isinstance(info, dict) else info
        x, y, w, h = bbox
        self.cx, self.cy = x + w / 2, y + h / 2
        self.w, self.h = w, h
        self._init_w = w
        self._init_h = h
        self._frame_count = 0
        self._hanning = None

        tmpl, _, _ = _crop(frame, self.cx, self.cy, w * h,
                           self.template_size, self.template_factor)

        if self.use_ort or self.use_trt:
            self.template_np = _preprocess_np(tmpl, self.template_size)
        else:
            self.template_tensor = _preprocess(tmpl, self.template_size).to(self.device)

    # =========================================================================
    # Track (dispatcher + template update)
    # =========================================================================

    def track(self, frame):
        """Track one frame. Returns {'target_bbox': [x,y,w,h], 'best_score': float}"""
        wh_area = self.w * self.h
        search_img, rf, crop_sz = _crop(frame, self.cx, self.cy, wh_area,
                                         self.search_size, self.search_factor)

        if self.use_trt:
            result = self._track_trt(search_img, rf)
        elif self.use_ort:
            result = self._track_ort(search_img, rf)
        else:
            result = self._track_pytorch(search_img, rf)

        # Periodic template update
        self._frame_count += 1
        if self._frame_count % 20 == 0 and result['best_score'] > 0.65:
            tmpl, _, _ = _crop(frame, self.cx, self.cy, self.w * self.h,
                               self.template_size, self.template_factor)
            if self.use_ort or self.use_trt:
                self.template_np = _preprocess_np(tmpl, self.template_size)
            else:
                self.template_tensor = _preprocess(tmpl, self.template_size).to(self.device)

        return result

    # =========================================================================
    # ONNX Runtime backend
    # =========================================================================

    def _track_ort(self, search_img, rf):
        search_np = _preprocess_np(search_img, self.search_size)

        score_map, size_map, offset_map = self.sess.run(
            None, {'template': self.template_np, 'search': search_np})

        sm = score_map[0, 0]
        self._ensure_hanning(sm.shape)

        mx, my, max_score, pw, ph, ox, oy = _decode_score_map(
            sm, self._hanning, self.search_size, size_map, offset_map)

        _update_state(self, mx, my, ox, oy, pw, ph, rf)

        return {
            'target_bbox': [self.cx - self.w / 2, self.cy - self.h / 2, self.w, self.h],
            'best_score': max_score,
        }

    # =========================================================================
    # TensorRT backend
    # =========================================================================

    def _track_trt(self, search_img, rf):
        search_np = _preprocess_np(search_img, self.search_size)
        search_t = torch.from_numpy(search_np).cuda()
        tmpl_t = torch.from_numpy(self.template_np).cuda()

        with torch.cuda.stream(self.cuda_stream):
            self.trt_inputs['template'].copy_(tmpl_t)
            self.trt_inputs['search'].copy_(search_t)
            self.trt_context.execute_async_v3(stream_handle=self.cuda_stream.cuda_stream)
        self.cuda_stream.synchronize()

        sm = self.trt_outputs['score_map'].cpu().numpy()[0, 0]
        size_map = self.trt_outputs['size_map'].cpu().numpy()
        offset_map = self.trt_outputs['offset_map'].cpu().numpy()

        self._ensure_hanning(sm.shape)

        mx, my, max_score, pw, ph, ox, oy = _decode_score_map(
            sm, self._hanning, self.search_size, size_map, offset_map)

        _update_state(self, mx, my, ox, oy, pw, ph, rf)

        return {
            'target_bbox': [self.cx - self.w / 2, self.cy - self.h / 2, self.w, self.h],
            'best_score': max_score,
        }

    # =========================================================================
    # PyTorch backend
    # =========================================================================

    def _track_pytorch(self, search_img, rf):
        search_tensor = _preprocess(search_img, self.search_size).to(self.device)

        with torch.no_grad(), torch.cuda.amp.autocast():
            x, _ = self.model.backbone(z=self.template_tensor, x=search_tensor)
            feat = x[-1] if isinstance(x, list) else x
            enc_opt = feat[:, -self.model.feat_len_s:]
            opt = enc_opt.unsqueeze(-1).permute(0, 3, 2, 1).contiguous()
            bs, Nq, C, HW = opt.size()
            opt_feat = opt.view(-1, C, self.model.feat_sz_s, self.model.feat_sz_s)
            score_map, bbox_pred, size_map, offset_map = self.model.box_head(opt_feat, None)

        sm = score_map.cpu().numpy()[0, 0]
        size_map_np = size_map.cpu().numpy()
        offset_map_np = offset_map.cpu().numpy()

        self._ensure_hanning(sm.shape)

        mx, my, max_score, pw, ph, ox, oy = _decode_score_map(
            sm, self._hanning, self.search_size, size_map_np, offset_map_np)

        _update_state(self, mx, my, ox, oy, pw, ph, rf)

        return {
            'target_bbox': [self.cx - self.w / 2, self.cy - self.h / 2, self.w, self.h],
            'best_score': max_score,
        }

    # =========================================================================
    # Helpers
    # =========================================================================

    def _ensure_hanning(self, sm_shape):
        """Create hanning window on first call or if score map size changes."""
        if self._hanning is None or self._hanning.shape != sm_shape:
            h, w = sm_shape
            self._hanning = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
            self._sm_size = h