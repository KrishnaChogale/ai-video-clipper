"""Robust facecam detection, persistent tracking, and reaction-proxy helpers.

Uses MediaPipe Face Landmarker as the primary engine for high-precision,
multi-landmark detection and FACS blendshape analysis, while retaining OpenCV
Haar Cascades as an automatic fallback for missing models or inference errors.
Implements Adaptive Velocity EMA smoothing to eliminate camera framing jitter.
"""
from __future__ import annotations

import logging
import math
import os
import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Default base directory resolution
try:
    import config
    BASE_DIR = getattr(config, "BASE_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
except ImportError:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "models", "face_landmarker.task")


class FacecamAnalyzer:
    """Detect and track small, persistent edge-positioned streamer facecams.
    
    Primary engine: MediaPipe Face Landmarker (CPU via TFLite XNNPACK delegate).
    Fallback engine: OpenCV Haar Cascades (frontalface + smile).
    Smoothing: Adaptive Velocity Exponential Moving Average (EMA).
    """

    def __init__(self, model_path: str | None = None, enable_smoothing: bool = True):
        self.model_path = model_path or DEFAULT_MODEL_PATH
        self.enable_smoothing = enable_smoothing
        self.engine = "haar"
        self.mp_detector = None
        self._mp = None
        self.face_cascade = None
        self.smile_cascade = None

        # Tracking and smoothing state
        self.previous_box: list[float] | None = None
        self.smoothed_box: list[float] | None = None
        self.previous_gray_face: np.ndarray | None = None
        self.lost_frames_count: int = 0
        self.max_lost_frames: int = 3  # Grace period: keep previous box for up to 3 frames

        # Initialize primary (MediaPipe) and fallback (Haar)
        self._init_haar()
        self._init_mediapipe(self.model_path)

    def _init_haar(self) -> bool:
        """Initialize OpenCV Haar cascades as dependable fallback."""
        try:
            cascade_cls = getattr(cv2, "CascadeClassifier", None)
            haar_dir = getattr(getattr(cv2, "data", None), "haarcascades", None)
            if cascade_cls and haar_dir:
                face_path = os.path.join(haar_dir, "haarcascade_frontalface_default.xml")
                if os.path.isfile(face_path):
                    face = cascade_cls(face_path)
                    if not face.empty():
                        self.face_cascade = face
                smile_path = os.path.join(haar_dir, "haarcascade_smile.xml")
                if os.path.isfile(smile_path):
                    smile = cascade_cls(smile_path)
                    if not smile.empty():
                        self.smile_cascade = smile
                return self.face_cascade is not None
        except Exception as e:
            logger.warning("Failed to initialize Haar cascades fallback: %s", e)
            self.face_cascade = None
        return False

    def _init_mediapipe(self, model_path: str) -> bool:
        """Initialize MediaPipe Face Landmarker."""
        if not os.path.isfile(model_path):
            logger.info("MediaPipe model asset not found at '%s'. Using Haar fallback.", model_path)
            self.engine = "haar"
            return False

        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=False,
                num_faces=3,  # Detect up to 3 faces to disambiguate streamer vs gameplay
                min_face_detection_confidence=0.35,
                min_face_presence_confidence=0.35,
                min_tracking_confidence=0.35,
            )
            self.mp_detector = vision.FaceLandmarker.create_from_options(options)
            self._mp = mp
            self.engine = "mediapipe"
            logger.info("MediaPipe Face Landmarker initialized successfully (engine=mediapipe).")
            return True
        except Exception as e:
            logger.warning("MediaPipe Face Landmarker initialization failed: %s. Falling back to Haar.", e)
            self.mp_detector = None
            self.engine = "haar"
            return False

    def reset(self) -> None:
        """Reset internal tracking and smoothing state."""
        self.previous_box = None
        self.smoothed_box = None
        self.previous_gray_face = None
        self.lost_frames_count = 0

    def analyze(self, frame: np.ndarray, inset_only: bool = True) -> dict:
        """Return a likely inset-face box and motion/smile/blendshape proxies.
        
        Args:
            frame: Input image (BGR format).
            inset_only: If True, filters out large cinematic faces and central characters
                        to specifically isolate streamer webcams.
                        
        Returns:
            dict containing:
              - box: [x, y, w, h] or None (smoothed)
              - raw_box: [x, y, w, h] or None (unsmoothed)
              - motion: float (0.0 - 100.0)
              - smile: float (0.0 - 100.0)
              - engine: 'mediapipe' or 'haar'
              - blendshapes: dict of FACS expressions (if MediaPipe active)
        """
        empty_result = {
            "box": None,
            "raw_box": None,
            "motion": 0.0,
            "smile": 0.0,
            "engine": self.engine,
            "blendshapes": {},
        }
        if frame is None or frame.size == 0:
            return empty_result

        height, width = frame.shape[:2]
        if height < 10 or width < 10:
            return empty_result

        # 1. Detection attempt
        raw_candidates = []
        active_engine = self.engine

        if self.engine == "mediapipe" and self.mp_detector is not None:
            try:
                raw_candidates = self._detect_mediapipe(frame)
            except Exception as e:
                logger.debug("MediaPipe inference error (%s); trying Haar fallback for frame", e)
                active_engine = "haar"
                raw_candidates = self._detect_haar(frame)
        else:
            active_engine = "haar"
            raw_candidates = self._detect_haar(frame)

        # 2. Filter candidates for streamer webcam characteristics
        filtered = []
        for cand in raw_candidates:
            x, y, w, h = cand["box"]
            cx, cy = x + w / 2.0, y + h / 2.0
            if inset_only:
                # Inset webcams are smaller than full-screen cinematic close-ups and near edges
                edge_dist = min(cx / width, 1.0 - cx / width, cy / height, 1.0 - cy / height)
                if (w / width > 0.45) or (h / height > 0.55) or (edge_dist > 0.38):
                    # Reject large cinematic center subjects unless no previous box exists
                    if self.previous_box is None:
                        continue
                    # If we are already tracking this subject, allow it even if slightly inward
                    px, py, pw, ph = self.previous_box
                    if math.hypot(cx - (px + pw / 2.0), cy - (py + ph / 2.0)) > width * 0.2:
                        continue
            filtered.append(cand)

        # 3. Handle detection absence / grace period
        if not filtered:
            self.lost_frames_count += 1
            if self.lost_frames_count <= self.max_lost_frames and self.smoothed_box is not None:
                # Grace period: hold previous smoothed box so framing doesn't jump
                return {
                    "box": [round(float(v), 2) for v in self.smoothed_box],
                    "raw_box": None,
                    "motion": 0.0,
                    "smile": 0.0,
                    "engine": active_engine,
                    "blendshapes": {},
                    "held_frame": True,
                }
            self.reset()
            return empty_result

        # Reset lost frame counter on successful detection
        self.lost_frames_count = 0

        # 4. Multi-face disambiguation: Pick persistent streamer face
        chosen = self._select_persistent_face(filtered, width, height)
        raw_box = chosen["box"]
        x, y, w, h = raw_box

        # 5. Adaptive Velocity EMA Smoothing
        if self.enable_smoothing:
            smoothed = self._apply_adaptive_ema(raw_box, width, height)
        else:
            smoothed = [float(v) for v in raw_box]
        self.smoothed_box = smoothed

        # 6. Motion Proxy Calculation
        face_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[int(y):int(y + h), int(x):int(x + w)]
        if face_gray.size > 0:
            face_gray = cv2.resize(face_gray, (48, 48))
        motion = 0.0
        if self.previous_gray_face is not None and face_gray.size > 0:
            diff = cv2.absdiff(face_gray, self.previous_gray_face)
            motion = min(100.0, float(diff.mean()) * 2.2)
        self.previous_gray_face = face_gray.copy() if face_gray.size > 0 else None
        self.previous_box = [float(v) for v in raw_box]

        # 7. Expression / Smile Proxy
        smile = float(chosen.get("smile", 0.0))
        blendshapes = chosen.get("blendshapes", {})

        return {
            "box": [round(float(v), 2) for v in smoothed],
            "raw_box": [round(float(v), 2) for v in raw_box],
            "motion": round(motion, 2),
            "smile": round(smile, 2),
            "engine": active_engine,
            "blendshapes": blendshapes,
        }

    def _detect_mediapipe(self, frame: np.ndarray) -> list[dict]:
        """Run MediaPipe Face Landmarker on a frame."""
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self.mp_detector.detect(mp_img)

        candidates = []
        if not result.face_landmarks:
            return candidates

        for idx, landmarks in enumerate(result.face_landmarks):
            xs = [lm.x * w for lm in landmarks]
            ys = [lm.y * h for lm in landmarks]
            xmin = max(0.0, min(xs))
            xmax = min(float(w), max(xs))
            ymin = max(0.0, min(ys))
            ymax = min(float(h), max(ys))
            box_w = max(1.0, xmax - xmin)
            box_h = max(1.0, ymax - ymin)

            # Extract FACS blendshapes
            blendshapes = {}
            smile_score = 0.0
            if result.face_blendshapes and idx < len(result.face_blendshapes):
                for b in result.face_blendshapes[idx]:
                    blendshapes[b.category_name] = round(float(b.score), 4)
                smile_l = blendshapes.get("mouthSmileLeft", 0.0)
                smile_r = blendshapes.get("mouthSmileRight", 0.0)
                smile_score = min(100.0, max(smile_l, smile_r) * 100.0)

            candidates.append({
                "box": [int(round(xmin)), int(round(ymin)), int(round(box_w)), int(round(box_h))],
                "smile": smile_score,
                "blendshapes": blendshapes,
            })
        return candidates

    def _detect_haar(self, frame: np.ndarray) -> list[dict]:
        """Run OpenCV Haar Cascade detection on a frame."""
        if self.face_cascade is None:
            return []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        try:
            boxes = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24)
            )
        except Exception:
            return []

        candidates = []
        for x, y, w, h in boxes:
            smile_score = 0.0
            if self.smile_cascade is not None:
                lower = gray[int(y + h * 0.42):int(y + h), int(x):int(x + w)]
                if lower.size > 0:
                    try:
                        smiles = self.smile_cascade.detectMultiScale(
                            lower, scaleFactor=1.7, minNeighbors=20, minSize=(max(12, int(w // 5)), 8)
                        )
                        smile_score = min(100.0, 55.0 + 15.0 * len(smiles)) if len(smiles) else 0.0
                    except Exception:
                        smile_score = 0.0
            candidates.append({
                "box": [int(x), int(y), int(w), int(h)],
                "smile": smile_score,
                "blendshapes": {},
            })
        return candidates

    def _select_persistent_face(self, candidates: list[dict], width: int, height: int) -> dict:
        """Select the most persistent streamer face from candidate detections."""
        if len(candidates) == 1:
            return candidates[0]

        if self.previous_box is not None:
            px, py, pw, ph = self.previous_box
            pcx, pcy = px + pw / 2.0, py + ph / 2.0
            p_area = max(1.0, pw * ph)
            diag = math.hypot(width, height)

            def tracking_cost(cand: dict) -> float:
                cx = cand["box"][0] + cand["box"][2] / 2.0
                cy = cand["box"][1] + cand["box"][3] / 2.0
                area = max(1.0, cand["box"][2] * cand["box"][3])
                dist_norm = math.hypot(cx - pcx, cy - pcy) / diag
                scale_ratio = max(area / p_area, p_area / area) - 1.0
                return dist_norm + 0.35 * min(2.0, scale_ratio)

            return min(candidates, key=tracking_cost)

        # If no previous box, prefer corner/edge inset faces over center
        def edge_preference(cand: dict) -> float:
            cx = cand["box"][0] + cand["box"][2] / 2.0
            cy = cand["box"][1] + cand["box"][3] / 2.0
            edge_dist = min(cx / width, 1.0 - cx / width, cy / height, 1.0 - cy / height)
            # Webcams are strictly closest to screen borders; minimize edge_dist
            return -edge_dist

        return max(candidates, key=edge_preference)

    def _apply_adaptive_ema(self, raw_box: list[float], width: int, height: int) -> list[float]:
        """Apply velocity-adaptive Exponential Moving Average smoothing on coordinates."""
        if self.smoothed_box is None:
            return [float(v) for v in raw_box]

        sx, sy, sw, sh = self.smoothed_box
        rx, ry, rw, rh = [float(v) for v in raw_box]

        scx, scy = sx + sw / 2.0, sy + sh / 2.0
        rcx, rcy = rx + rw / 2.0, ry + rh / 2.0

        disp = math.hypot(rcx - scx, rcy - scy)
        # Deadzone: 2.5 pixels (on 640x360 frame) or 0.7% of diagonal
        deadzone = max(2.5, math.hypot(width, height) * 0.006)

        if disp <= deadzone:
            # Stationary / micro-jitter: heavy smoothing (alpha = 0.08)
            alpha_pos = 0.08
        else:
            # Dynamic response up to alpha = 0.60 based on velocity
            factor = min(1.0, (disp - deadzone) / max(1.0, deadzone * 8.0))
            alpha_pos = 0.15 + factor * 0.45

        # Size smoothing (conservative to avoid breathing artifacts)
        alpha_size = 0.12

        new_cx = alpha_pos * rcx + (1.0 - alpha_pos) * scx
        new_cy = alpha_pos * rcy + (1.0 - alpha_pos) * scy
        new_w = alpha_size * rw + (1.0 - alpha_size) * sw
        new_h = alpha_size * rh + (1.0 - alpha_size) * sh

        new_x = max(0.0, min(float(width) - new_w, new_cx - new_w / 2.0))
        new_y = max(0.0, min(float(height) - new_h, new_cy - new_h / 2.0))

        return [new_x, new_y, new_w, new_h]


def summarize_facecam(visual_scores: list[dict], source_width: int, source_height: int) -> dict:
    """Find the most persistent inset-face location and per-second reaction proxies.
    
    Uses spatial density clustering across valid observations to isolate the streamer's
    webcam inset from spurious game detections.
    """
    observations = []
    for index, item in enumerate(visual_scores or []):
        box = item.get("facecam_box")
        if box and len(box) == 4:
            observations.append((index, [float(v) for v in box]))

    empty = {
        "available": False,
        "region": None,
        "presence": [],
        "reaction": [],
        "coverage": 0.0,
        "observations": 0,
    }
    if len(observations) < 3:
        return empty

    # Density clustering: cluster observations within spatial radius
    diag = math.hypot(640.0, 360.0)
    clusters = []
    for obs in observations:
        x, y, w, h = obs[1]
        cx, cy = x + w / 2.0, y + h / 2.0
        placed = False
        for c in clusters:
            ccx, ccy = c["center"]
            if math.hypot(cx - ccx, cy - ccy) < diag * 0.12:  # within ~85px on 640x360
                c["items"].append(obs)
                # Update running center
                all_cx = [it[1][0] + it[1][2] / 2.0 for it in c["items"]]
                all_cy = [it[1][1] + it[1][3] / 2.0 for it in c["items"]]
                c["center"] = (float(np.mean(all_cx)), float(np.mean(all_cy)))
                placed = True
                break
        if not placed:
            clusters.append({"center": (cx, cy), "items": [obs]})

    if not clusters:
        return empty

    # Select the most populated cluster (streamer webcam)
    best_cluster_obj = max(clusters, key=lambda c: len(c["items"]))
    best_cluster = best_cluster_obj["items"]

    min_observations = max(3, min(20, int(len(visual_scores) * 0.001)))
    if len(best_cluster) < min_observations:
        return empty

    boxes = np.asarray([item[1] for item in best_cluster], dtype=np.float32)
    # Use trimmed/median coordinates for robust central position
    x, y, w, h = np.median(boxes, axis=0).tolist()
    if source_width <= 0 or source_height <= 0:
        return empty

    # Size top panel crop around facecam with room for head, shoulders, and lighting
    panel_aspect = 1080.0 / 864.0
    crop_h = min(float(source_height) * 0.65, max(h * source_height / 360.0 * 1.5, source_height * 0.15))
    crop_w = min(float(source_width), crop_h * panel_aspect)
    crop_h = crop_w / panel_aspect
    face_x = (x + w / 2.0) * source_width / 640.0
    face_y = (y + h / 2.0) * source_height / 360.0

    region = {
        "width": int(round(crop_w)),
        "height": int(round(crop_h)),
        "x": int(round(max(0.0, min(source_width - crop_w, face_x - crop_w / 2.0)))),
        "y": int(round(max(0.0, min(source_height - crop_h, face_y - crop_h / 2.0)))),
    }

    presence = [0.0] * len(visual_scores)
    reaction = [0.0] * len(visual_scores)

    for index, box in best_cluster:
        item = visual_scores[index]
        presence[index] = 1.0
        # Combine motion, smile proxy, and blendshape cues
        mot = float(item.get("facecam_motion", 0.0))
        sml = float(item.get("facecam_smile", 0.0))
        blendshapes = item.get("blendshapes") or {}
        jaw = float(blendshapes.get("jawOpen", 0.0)) * 100.0
        brow = float(blendshapes.get("browInnerUp", 0.0)) * 100.0

        reaction_score = 0.55 * mot + 0.25 * sml + 0.12 * jaw + 0.08 * brow
        reaction[index] = max(0.0, min(100.0, reaction_score))

    return {
        "available": True,
        "region": region,
        "presence": presence,
        "reaction": reaction,
        "coverage": round(len(best_cluster) / max(1, len(visual_scores)), 4),
        "observations": len(best_cluster),
    }


def build_facecam_layout(
    visual_scores: list[dict],
    start: float,
    end: float,
    source_width: int,
    source_height: int,
    reaction: dict,
    facecam_percent: int = 45,
    headline_text: str = "",
    branding_text: str = "",
    reaction_events: list | None = None,
    gameplay_events: list | None = None,
    zoom_enabled: bool = True,
    accent_color: str = "0x00F0FF",
    banner_bg: str = "0x16181C",
) -> dict | None:
    """Build tracked crop points and gameplay occlusion bounds for the selected window."""
    if not reaction.get("available"):
        return None

    region = reaction["region"]
    tracks = []
    boxes = []

    scores_by_sec = {}
    for idx, it in enumerate(visual_scores or []):
        s_val = it.get("second")
        if s_val is not None:
            scores_by_sec[int(s_val)] = (idx, it)
        else:
            scores_by_sec[idx] = (idx, it)

    for second in range(int(start), int(end) + 1):
        entry = scores_by_sec.get(second)
        if entry is None:
            continue
        idx, item = entry
        box = item.get("facecam_box")
        presence = reaction["presence"][idx] if idx < len(reaction.get("presence", [])) else 1.0
        if not box or not presence:
            continue
        x, y, w, h = box
        cx = (float(x) + float(w) / 2.0) * source_width / 640.0
        cy = (float(y) + float(h) / 2.0) * source_height / 360.0
        crop_w, crop_h = region["width"], region["height"]
        crop_x = max(0.0, min(source_width - crop_w, cx - crop_w / 2.0))
        crop_y = max(0.0, min(source_height - crop_h, cy - crop_h / 2.0))
        tracks.append({"t": max(0.0, second - start), "x": crop_x, "y": crop_y})

        scale_x, scale_y = source_width / 640.0, source_height / 360.0
        # Gameplay occlusion mask bounds
        boxes.append((
            max(0, int((float(x) - float(w) * 1.2) * scale_x)),
            max(0, int((float(y) - float(h) * 0.9) * scale_y)),
            min(source_width, int((float(x) + float(w) * 2.2) * scale_x)),
            min(source_height, int((float(y) + float(h) * 1.9) * scale_y)),
        ))

    if not tracks:
        # Graceful fallback: anchor to verified static region if temporal tracks are sparse
        tracks.append({"t": 0.0, "x": float(region["x"]), "y": float(region["y"])})
        tracks.append({"t": max(1.0, float(end - start)), "x": float(region["x"]), "y": float(region["y"])})

    if boxes:
        ox1 = int(np.median([b[0] for b in boxes]))
        oy1 = int(np.median([b[1] for b in boxes]))
        ox2 = int(np.median([b[2] for b in boxes]))
        oy2 = int(np.median([b[3] for b in boxes]))
    else:
        # Default overlay covering the region in gameplay
        ox1 = int(region["x"])
        oy1 = int(region["y"])
        ox2 = ox1 + int(region["width"])
        oy2 = oy1 + int(region["height"])

    face_zoom_expr = None
    game_zoom_expr = None
    if zoom_enabled and (reaction_events or gameplay_events):
        try:
            from modules.layout_engine import ReactionZoomController
            zoom_res = ReactionZoomController.build_zoom_expressions(
                reaction_events=reaction_events,
                gameplay_events=gameplay_events,
                start_sec=start,
                end_sec=end,
                enabled=True,
            )
            face_zoom_expr = zoom_res.get("face_zoom_expr")
            game_zoom_expr = zoom_res.get("game_zoom_expr")
        except Exception:
            face_zoom_expr = None
            game_zoom_expr = None

    return {
        "region": region,
        "tracks": tracks,
        "overlay": {
            "x": ox1,
            "y": oy1,
            "width": max(1, ox2 - ox1),
            "height": max(1, oy2 - oy1),
        },
        "facecam_percent": max(35, min(65, int(facecam_percent))),
        "headline_text": (headline_text or "")[:120],
        "branding_text": (branding_text or "")[:60],
        "face_zoom_expr": face_zoom_expr,
        "game_zoom_expr": game_zoom_expr,
        "accent_color": accent_color,
        "banner_bg": banner_bg,
    }
