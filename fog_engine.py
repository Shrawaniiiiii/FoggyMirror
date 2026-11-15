"""Fog state: breath-made fog, wiped areas, brush stamps, moisture rim, drips."""

import math

import cv2
import numpy as np

import config


def make_noise(h, w, rng, cell_sizes):
    """Smooth multi-octave noise, normalized to 0..1."""
    out = np.zeros((h, w), np.float32)
    amp, total = 1.0, 0.0
    for cell in cell_sizes:
        gh = max(2, h // cell + 2)
        gw = max(2, w // cell + 2)
        grid = rng.random((gh, gw)).astype(np.float32)
        out += cv2.resize(grid, (w, h), interpolation=cv2.INTER_CUBIC) * amp
        total += amp
        amp *= 0.5
    out /= total
    out -= out.min()
    out /= (out.max() + 1e-6)
    return out


def smoothstep(x):
    return x * x * (3.0 - 2.0 * x)


class BrushLibrary:
    """Pre-generated soft, irregular brush shapes for finger and palm."""

    BASE = 96
    VARIANTS = 8

    def __init__(self, seed=11):
        self.rng = np.random.default_rng(seed)
        B = self.BASE
        c = (B - 1) / 2.0
        ax = (np.arange(B, dtype=np.float32) - c) / c * 1.2   # covers -1.2 .. 1.2
        u, v = np.meshgrid(ax, ax)
        d = np.sqrt(u * u + v * v)

        self.sprites = {"finger": [], "palm": []}
        for _ in range(self.VARIANTS):
            # Finger: round core, soft edge, slightly ragged boundary.
            n = 0.6 * self._rand(6, 6) + 0.4 * self._rand(14, 14)
            edge = 1.0 - d + (n - 0.5) * 0.30
            s = smoothstep(np.clip(edge / 0.28, 0.0, 1.0))
            self.sprites["finger"].append(s.astype(np.float32))

            # Palm: wider soft edge with streaks inside.
            if self.rng.random() < 0.5:
                streak = self._rand(3, 16)
            else:
                streak = self._rand(16, 3)
            n = 0.5 * streak + 0.5 * self._rand(5, 5)
            edge = 1.0 - d + (n - 0.5) * 0.6
            s = smoothstep(np.clip(edge / 0.8, 0.0, 1.0)) * (0.85 + 0.15 * streak)
            self.sprites["palm"].append(s.astype(np.float32))

        self._cache = {}

    def _rand(self, gh, gw):
        grid = self.rng.random((gh, gw)).astype(np.float32)
        img = cv2.resize(grid, (self.BASE, self.BASE), interpolation=cv2.INTER_CUBIC)
        img -= img.min()
        img /= (img.max() + 1e-6)
        return img

    def get(self, kind, radius, variant):
        r = max(1, int(round(radius)))
        key = (kind, variant, r)
        spr = self._cache.get(key)
        if spr is None:
            size = 2 * int(math.ceil(r * 1.2)) + 1
            spr = cv2.resize(self.sprites[kind][variant], (size, size),
                             interpolation=cv2.INTER_AREA if size < self.BASE
                             else cv2.INTER_LINEAR).astype(np.float32)
            self._cache[key] = spr
        return spr


class FogEngine:
    def __init__(self, frame_w, frame_h):
        self.w = max(16, int(frame_w * config.FOG_SCALE))
        self.h = max(16, int(frame_h * config.FOG_SCALE))
        self.aspect = self.w / float(self.h)

        self.base = np.zeros((self.h, self.w), np.float32)      # fog from breath
        self.cleared = np.zeros((self.h, self.w), np.float32)   # wiped amount
        self.mask = np.zeros((self.h, self.w), np.float32)      # final fog 0..1
        self.wet = np.zeros((self.h, self.w), np.float32)       # wiped glass under fog
        self.rim = np.zeros((self.h, self.w), np.float32)       # stroke edge moisture

        self.density = config.FOG_DENSITY
        self.breath = 0.0
        self.center = (0.5, 0.65)

        self.rng = np.random.default_rng(7)
        self.noise = make_noise(self.h, self.w, self.rng, [40, 20, 10])
        self.regrow_noise = make_noise(self.h, self.w, self.rng, [30, 12])
        self.residual_noise = make_noise(self.h, self.w, self.rng, [6, 3])

        self.xs = np.linspace(0.0, 1.0, self.w, dtype=np.float32)[None, :]
        self.ys = np.linspace(0.0, 1.0, self.h, dtype=np.float32)[:, None]

        self.brushes = BrushLibrary()
        self._variant = 0
        self._vcount = 0

        self.drips = []     # each: [x, y, speed, remaining_length]
        self._drip_sprite = np.array([[0.0, 0.25, 0.0],
                                      [0.25, 1.0, 0.25],
                                      [0.0, 0.25, 0.0]], np.float32)

    # ------------------------------------------------------------------
    # Brush stamping
    # ------------------------------------------------------------------
    def _stamp(self, spr, cx, cy, strength, mode="max"):
        sh, sw = spr.shape
        x0 = int(round(cx)) - sw // 2
        y0 = int(round(cy)) - sh // 2
        x1, y1 = x0 + sw, y0 + sh
        ax0, ay0 = max(x0, 0), max(y0, 0)
        ax1, ay1 = min(x1, self.w), min(y1, self.h)
        if ax1 <= ax0 or ay1 <= ay0:
            return
        sub = spr[ay0 - y0:ay1 - y0, ax0 - x0:ax1 - x0] * strength
        roi = self.cleared[ay0:ay1, ax0:ax1]
        if mode == "max":
            np.maximum(roi, sub, out=roi)
        else:  # "screen": repeated passes keep clearing more
            roi[:] = 1.0 - (1.0 - roi) * (1.0 - sub)

    def stroke(self, kind, p0, p1, radius_px, strength):
        """Clear along the path p0 -> p1 (normalized coords). p0 may be None."""
        radius_px = max(1.0, radius_px)
        mode = "max" if kind == "finger" else "screen"
        x1, y1 = p1[0] * self.w, p1[1] * self.h

        if p0 is None:
            pts = [(x1, y1)]
        else:
            x0, y0 = p0[0] * self.w, p0[1] * self.h
            dist = math.hypot(x1 - x0, y1 - y0)
            step = max(1.0, radius_px * config.BRUSH_SPACING)
            n = max(1, int(math.ceil(dist / step)))
            pts = [(x0 + (x1 - x0) * i / n, y0 + (y1 - y0) * i / n)
                   for i in range(1, n + 1)]

        for (x, y) in pts:
            # Change the brush shape every few stamps -> natural ragged edge.
            self._vcount += 1
            if self._vcount >= 5:
                self._vcount = 0
                self._variant = int(self.rng.integers(BrushLibrary.VARIANTS))
            spr = self.brushes.get(kind, radius_px, self._variant)
            self._stamp(spr, x, y, strength, mode)

    # ------------------------------------------------------------------
    # Simulation step
    # ------------------------------------------------------------------
    def update(self, dt, mouth_open, mouth_center):
        if mouth_center is not None:
            self.center = mouth_center

        target = 1.0 if mouth_open else 0.0
        speed = config.BREATH_RISE_SPEED if target > self.breath \
            else config.BREATH_FALL_SPEED
        self.breath += (target - self.breath) * min(1.0, speed * dt)

        if self.breath > 0.01:
            cx, cy = self.center
            dx = (self.xs - cx) * self.aspect
            dy = self.ys - cy
            sigma = config.BREATH_SIGMA
            plume = np.exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma))
            g = config.FOG_SPREAD
            spread = plume * (1.0 - g) + g
            grain = 0.55 + 0.9 * self.noise
            self.base += dt * config.FOG_FORM_RATE * self.breath * spread * grain
            # Breath also re-fogs glass that was wiped earlier.
            self.cleared -= dt * config.REFOG_BY_BREATH * self.breath * spread

        self.base -= dt * config.FOG_FADE_RATE * (1.0 - self.breath)

        # Wiped areas slowly fog over again, in patches.
        self.cleared -= dt * config.FOG_REGROW_RATE * (0.4 + 1.2 * self.regrow_noise)

        np.clip(self.base, 0.0, 1.0, out=self.base)
        np.clip(self.cleared, 0.0, 1.0, out=self.cleared)

        if config.DRIPS_ENABLED:
            self._update_drips(dt)

        self._compose()

    def _compose(self):
        # Wiped glass keeps a trace of moisture (never perfectly clean).
        eff = self.cleared * (0.94 + 0.06 * self.residual_noise)
        self.mask = self.base * (1.0 - eff)
        self.wet = self.base * self.cleared

        # Moisture rim: where the wiped area meets the fog.
        g = cv2.GaussianBlur(self.cleared, (0, 0), 1.0)
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        self.rim = np.clip(mag * config.RIM_STRENGTH, 0.0, 1.0) * \
            np.clip(self.base * 1.5, 0.0, 1.0)

    def _update_drips(self, dt):
        if (len(self.drips) < config.DRIP_MAX
                and self.rng.random() < config.DRIP_SPAWN_RATE * dt):
            for _ in range(12):
                x = int(self.rng.integers(2, self.w - 2))
                y = int(self.rng.integers(2, self.h - 8))
                if self.rim[y, x] > 0.25 and self.base[y, x] > 0.5:
                    self.drips.append([float(x), float(y),
                                       float(self.rng.uniform(5.0, 12.0)),
                                       float(self.rng.uniform(10.0, 45.0))])
                    break

        alive = []
        for d in self.drips:
            step = d[2] * dt
            d[1] += step
            d[0] += float(self.rng.normal(0.0, 0.6)) * dt   # slight wobble
            d[3] -= step
            self._stamp(self._drip_sprite, d[0], d[1], 0.85, "max")
            if d[3] > 0 and d[1] < self.h - 2:
                alive.append(d)
        self.drips = alive

    # ------------------------------------------------------------------
    def clear(self):
        """Remove all fog (key C)."""
        self.base[:] = 0.0
        self.cleared[:] = 0.0
        self.drips = []

    def fog_all(self):
        """Instantly fully fog the mirror (key G, handy for testing)."""
        self.base[:] = 1.0
        self.cleared[:] = 0.0

    def adjust_density(self, delta):
        self.density = float(np.clip(self.density + delta, 0.3, 1.0))