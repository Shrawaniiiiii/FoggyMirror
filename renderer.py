"""Builds the final image: live camera + fog layer blended by the fog mask."""

import cv2
import numpy as np

import config
from fog_engine import make_noise


class FogRenderer:
    def __init__(self, frame_w, frame_h):
        w, h = frame_w, frame_h
        rng = np.random.default_rng(42)

        # Fine grain like frosted condensation.
        grain = rng.normal(0.0, 1.0, (h, w)).astype(np.float32)
        grain = cv2.GaussianBlur(grain, (0, 0), 0.8)

        # Larger condensation droplets.
        drops = np.zeros((h, w), np.float32)
        count = int(w * h / 1200)
        px = rng.integers(0, w, count)
        py = rng.integers(0, h, count)
        radii = rng.choice([1, 1, 1, 2, 2, 3], count)
        bright = rng.uniform(0.4, 1.0, count)
        for x, y, r, b in zip(px, py, radii, bright):
            cv2.circle(drops, (int(x), int(y)), int(r), float(b), -1, cv2.LINE_AA)
        drops = cv2.GaussianBlur(drops, (0, 0), 1.0)
        self.drops_n = np.clip(drops / (drops.max() + 1e-6), 0.0, 1.0)

        # Many tiny beads: the moisture that stays on wiped glass.
        beads = np.zeros((h, w), np.float32)
        count = int(w * h / 450)
        px = rng.integers(0, w, count)
        py = rng.integers(0, h, count)
        bright = rng.uniform(0.3, 1.0, count)
        for x, y, b in zip(px, py, bright):
            cv2.circle(beads, (int(x), int(y)), 1, float(b), -1, cv2.LINE_AA)
        beads = cv2.GaussianBlur(beads, (0, 0), 0.7)
        self.beads_n = np.clip(beads / (beads.max() + 1e-6), 0.0, 1.0)

        smudge = make_noise(h, w, rng, [160, 80, 40])

        # Combined texture added on top of the fog color (2D, no channel axis).
        self.texture = (0.025 * grain + 0.08 * self.drops_n
                        + 0.07 * (smudge - 0.5)).astype(np.float32)

        self.alpha_noise = make_noise(h, w, rng, [90, 45, 20])

        # Fog is slightly denser toward the edges of the mirror.
        ys = np.linspace(-1.0, 1.0, h, dtype=np.float32)[:, None]
        xs = np.linspace(-1.0, 1.0, w, dtype=np.float32)[None, :]
        r2 = (xs * xs + ys * ys) / 2.0
        self.vignette = ((1.0 - config.VIGNETTE / 2.0) + config.VIGNETTE * r2) \
            .astype(np.float32)

        tint = np.array(config.FOG_TINT_BGR, np.float32) / 255.0
        self.tint = tint[None, None, :]

    def render(self, frame, fog, density):
        h, w = frame.shape[:2]

        # 1) Blurred grayscale scene seen through the fog.
        small = cv2.resize(frame, (w // 8, h // 8), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (0, 0), config.FOG_BLUR_SIGMA)
        gray = cv2.resize(gray, (w, h), interpolation=cv2.INTER_LINEAR)
        gray = gray.astype(np.float32) / 255.0

        # 2) Upscale the simulation maps to full resolution.
        m = cv2.resize(fog.mask, (w, h), interpolation=cv2.INTER_LINEAR)
        rim = cv2.resize(fog.rim, (w, h), interpolation=cv2.INTER_LINEAR)
        wet = cv2.resize(fog.wet, (w, h), interpolation=cv2.INTER_LINEAR)
        m = np.clip(m, 0.0, 1.0)
        m = m * m * (3.0 - 2.0 * m)                      # soft transitions

        # 3) Fog opacity: base fog + moisture rim + beads left on wiped glass.
        alpha = m * density * (0.85 + 0.3 * self.alpha_noise) * self.vignette
        alpha += rim * config.RIM_ALPHA
        alpha += wet * self.beads_n * config.RESIDUAL_DROPLETS
        np.clip(alpha, 0.0, 1.0, out=alpha)

        # 4) Fog color: tinted blurred scene + texture + bright moisture rim.
        floor = config.FOG_BRIGHTNESS_FLOOR
        bright = floor + (1.0 - floor) * gray
        tex = self.texture + rim * (config.RIM_BRIGHT + 0.25 * self.drops_n)
        fog_color = self.tint * bright[:, :, None] + tex[:, :, None]

        # 5) Blend: clear glass where alpha = 0, fog where alpha = 1.
        clear = frame.astype(np.float32) * (1.0 / 255.0)
        out = clear + (fog_color - clear) * alpha[:, :, None]
        return (np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8)