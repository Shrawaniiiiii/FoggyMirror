"""Face tracking with MediaPipe FaceMesh: mouth-open detection."""

from dataclasses import dataclass, field

import cv2
import mediapipe as mp
import numpy as np

import config

# FaceMesh landmark indices
UPPER_LIP = 13      # upper inner lip, center
LOWER_LIP = 14      # lower inner lip, center
MOUTH_LEFT = 61     # mouth corner
MOUTH_RIGHT = 291   # mouth corner


@dataclass
class FaceState:
    found: bool = False
    ratio: float = 0.0                       # smoothed mouth-open ratio
    is_open: bool = False                    # True when the mouth is open
    mouth_center: tuple = (0.5, 0.65)        # normalized (x, y) of the mouth
    mouth_points: list = field(default_factory=list)  # normalized points for debug


class FaceTracker:
    def __init__(self):
        self.mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,   # we do not need iris landmarks -> faster
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.state = FaceState()
        self._count = 0

    def process(self, frame):
        """Analyse a BGR frame and return the current FaceState."""
        run_now = (self._count % config.FACE_EVERY_N_FRAMES) == 0
        self._count += 1
        if not run_now:
            return self.state

        h, w = frame.shape[:2]
        scale = config.TRACK_WIDTH / float(w)
        sw, sh = config.TRACK_WIDTH, int(h * scale)
        small = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA)

        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False          # lets MediaPipe skip a copy
        result = self.mesh.process(rgb)

        st = self.state
        if not result.multi_face_landmarks:
            st.found = False
            st.is_open = False
            st.ratio *= 0.8                   # fade the ratio out
            st.mouth_points = []
            return st

        lm = result.multi_face_landmarks[0].landmark

        def pt(i):
            # Convert to pixel units so the aspect ratio is correct.
            return np.array([lm[i].x * sw, lm[i].y * sh])

        height = np.linalg.norm(pt(UPPER_LIP) - pt(LOWER_LIP))
        width = np.linalg.norm(pt(MOUTH_LEFT) - pt(MOUTH_RIGHT))
        raw_ratio = height / max(width, 1e-6)

        a = config.MOUTH_RATIO_SMOOTHING
        st.ratio = a * raw_ratio + (1.0 - a) * st.ratio

        # Hysteresis: different thresholds for opening and closing.
        if st.is_open:
            if st.ratio < config.MOUTH_OPEN_OFF:
                st.is_open = False
        else:
            if st.ratio > config.MOUTH_OPEN_ON:
                st.is_open = True

        cx = (lm[UPPER_LIP].x + lm[LOWER_LIP].x) / 2.0
        cy = (lm[UPPER_LIP].y + lm[LOWER_LIP].y) / 2.0
        # Light smoothing of the position so the fog plume does not jitter.
        st.mouth_center = (0.6 * st.mouth_center[0] + 0.4 * cx,
                           0.6 * st.mouth_center[1] + 0.4 * cy)

        st.found = True
        st.mouth_points = [(lm[i].x, lm[i].y) for i in
                           (UPPER_LIP, LOWER_LIP, MOUTH_LEFT, MOUTH_RIGHT)]
        return st

    def close(self):
        self.mesh.close()