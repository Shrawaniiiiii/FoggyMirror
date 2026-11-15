"""Turns hand gestures into strokes on the fog (finger writing, palm wiping)."""

import math

import numpy as np

import config


class Interaction:
    def __init__(self, fog, frame_w, frame_h):
        self.fog = fog
        self.aspect = frame_w / float(frame_h)
        self.prev_finger = None
        self.prev_palm = None
        self.lost_finger = 0
        self.lost_palm = 0

    def _dist(self, a, b):
        """Distance in frame-heights between two normalized points."""
        return math.hypot((a[0] - b[0]) * self.aspect, a[1] - b[1])

    def update(self, hand):
        finger_active = hand.found and hand.gesture == "index"
        palm_active = hand.found and hand.gesture == "palm"

        if finger_active:
            self._finger(hand)
        else:
            self.lost_finger += 1
            if self.lost_finger > config.STROKE_LOST_GRACE:
                self.prev_finger = None

        if palm_active:
            self._palm(hand)
        else:
            self.lost_palm += 1
            if self.lost_palm > config.STROKE_LOST_GRACE:
                self.prev_palm = None

    def _finger(self, hand):
        p = hand.index_tip
        if self.prev_finger is not None and \
                self._dist(p, self.prev_finger) > config.STROKE_JUMP_LIMIT:
            self.prev_finger = None          # tracking jump: start a new stroke

        r = float(np.clip(hand.hand_size * config.FINGER_RADIUS_RATIO,
                          config.FINGER_RADIUS_MIN, config.FINGER_RADIUS_MAX))
        r *= 1.0 - 0.15 * min(hand.speed / 1.5, 1.0)   # fast = slightly thinner
        self.fog.stroke("finger", self.prev_finger, p,
                        r * self.fog.h, config.FINGER_STRENGTH)
        self.prev_finger = p
        self.lost_finger = 0

    def _palm(self, hand):
        c = hand.palm_center
        if self.prev_palm is not None and \
                self._dist(c, self.prev_palm) > config.STROKE_JUMP_LIMIT:
            self.prev_palm = None

        # Only wipe when the palm is really moving.
        if self.prev_palm is not None and hand.speed > config.PALM_MIN_SPEED:
            r = float(np.clip(hand.hand_size * config.PALM_RADIUS_RATIO,
                              config.PALM_RADIUS_MIN, config.PALM_RADIUS_MAX))
            strength = config.PALM_STAMP_STRENGTH * \
                (0.6 + 0.4 * min(hand.speed / 1.2, 1.0))
            self.fog.stroke("palm", self.prev_palm, c, r * self.fog.h, strength)
        self.prev_palm = c
        self.lost_palm = 0