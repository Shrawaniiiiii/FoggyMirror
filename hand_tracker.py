"""Hand tracking with MediaPipe Hands: gesture classification and smoothing."""

import math
from dataclasses import dataclass, field

import cv2
import mediapipe as mp
import numpy as np

import config

HAND_CONNECTIONS = mp.solutions.hands.HAND_CONNECTIONS

# (tip, pip) landmark pairs for index, middle, ring, pinky
FINGER_JOINTS = [(8, 6), (12, 10), (16, 14), (20, 18)]
PALM_POINTS = [0, 5, 9, 13, 17]


class OneEuro:
    """One Euro filter: smooth when slow, responsive when fast."""

    def __init__(self, min_cutoff, beta, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = 0.0

    def reset(self):
        self.x_prev = None
        self.dx_prev = 0.0

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, dt):
        if self.x_prev is None:
            self.x_prev = x
            return x
        dt = max(dt, 1e-3)
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1.0 - a) * self.x_prev
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        return x_hat


class Filter2D:
    def __init__(self):
        self.fx = OneEuro(config.ONE_EURO_MIN_CUTOFF, config.ONE_EURO_BETA)
        self.fy = OneEuro(config.ONE_EURO_MIN_CUTOFF, config.ONE_EURO_BETA)

    def reset(self):
        self.fx.reset()
        self.fy.reset()

    def __call__(self, p, dt):
        return (self.fx(p[0], dt), self.fy(p[1], dt))


@dataclass
class HandState:
    found: bool = False
    gesture: str = "none"                 # "none" | "index" | "palm"
    index_tip: tuple = (0.5, 0.5)         # normalized, smoothed
    palm_center: tuple = (0.5, 0.5)       # normalized, smoothed
    speed: float = 0.0                    # palm speed in frame-heights / second
    hand_size: float = 0.2                # wrist -> middle knuckle, in frame heights
    fingers: tuple = (False, False, False, False)
    points: list = field(default_factory=list)   # 21 normalized points (debug)


class HandTracker:
    def __init__(self):
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=config.HAND_MODEL_COMPLEXITY,
            min_detection_confidence=config.HAND_DETECTION_CONF,
            min_tracking_confidence=config.HAND_TRACKING_CONF,
        )
        self.state = HandState()
        self._tip_f = Filter2D()
        self._palm_f = Filter2D()
        self._t_prev = None
        self._was_found = False
        self._pending = None
        self._pending_count = 0

    # ---- gesture debouncing ----
    def _set_gesture(self, raw):
        st = self.state
        if raw == st.gesture:
            self._pending = None
            self._pending_count = 0
            return
        if raw == self._pending:
            self._pending_count += 1
        else:
            self._pending = raw
            self._pending_count = 1
        if self._pending_count >= config.GESTURE_HOLD_FRAMES:
            st.gesture = raw
            self._pending = None
            self._pending_count = 0

    def process(self, frame, t):
        """Analyse a BGR frame taken at time t (seconds). Returns HandState."""
        dt = 1.0 / 30.0 if self._t_prev is None else max(t - self._t_prev, 1e-3)
        self._t_prev = t

        h, w = frame.shape[:2]
        sw = config.HAND_TRACK_WIDTH
        sh = int(h * sw / float(w))
        small = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = self.hands.process(rgb)

        st = self.state
        if not result.multi_hand_landmarks:
            st.found = False
            st.speed *= 0.5
            st.points = []
            self._tip_f.reset()
            self._palm_f.reset()
            self._was_found = False
            self._set_gesture("none")
            return st

        lm = result.multi_hand_landmarks[0].landmark
        P = np.array([[p.x * sw, p.y * sh] for p in lm], np.float32)  # pixels

        wrist = P[0]

        def extended(tip, pip):
            d_tip = np.linalg.norm(P[tip] - wrist)
            d_pip = np.linalg.norm(P[pip] - wrist)
            return bool(d_tip > d_pip * config.EXTEND_RATIO)

        fingers = tuple(extended(t_, p_) for t_, p_ in FINGER_JOINTS)
        if fingers[0] and not any(fingers[1:]):
            raw = "index"
        elif all(fingers):
            raw = "palm"
        else:
            raw = "none"

        # ---- positions (normalized) ----
        tip = (float(P[8][0] / sw), float(P[8][1] / sh))
        pc = P[PALM_POINTS].mean(axis=0)
        palm = (float(pc[0] / sw), float(pc[1] / sh))

        tip_s = self._tip_f(tip, dt)
        palm_s = self._palm_f(palm, dt)

        # ---- palm speed in frame-heights per second ----
        aspect = sw / float(sh)
        if self._was_found:
            dx = (palm_s[0] - st.palm_center[0]) * aspect
            dy = palm_s[1] - st.palm_center[1]
            raw_speed = math.hypot(dx, dy) / dt
            st.speed = 0.5 * raw_speed + 0.5 * st.speed
        else:
            st.speed = 0.0

        size = float(np.linalg.norm(P[9] - wrist) / sh)
        st.hand_size = 0.8 * st.hand_size + 0.2 * size if self._was_found else size

        st.found = True
        st.index_tip = tip_s
        st.palm_center = palm_s
        st.fingers = fingers
        st.points = [(p.x, p.y) for p in lm]
        self._was_found = True
        self._set_gesture(raw)
        return st

    def close(self):
        self.hands.close()