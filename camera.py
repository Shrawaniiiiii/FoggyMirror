"""Webcam wrapper: opens the camera, reads frames, mirrors them."""

import cv2

import config


class Camera:
    def __init__(self):
        # CAP_DSHOW is much faster to open on Windows than the default backend.
        self.cap = cv2.VideoCapture(config.CAMERA_INDEX, cv2.CAP_DSHOW)

        if not self.cap.isOpened():
            raise RuntimeError(
                "Could not open the webcam. Check that it is connected, "
                "not used by another app (Zoom/Teams), and that "
                "CAMERA_INDEX in config.py is correct."
            )

        # The format must be set BEFORE the size, otherwise some cameras ignore it.
        if config.USE_MJPG:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, config.TARGET_FPS)
        # Keep only the newest frame in the buffer to reduce lag.
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # The camera may not support the requested size, so read the real one.
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def read(self):
        """Return the next frame (BGR image) or None if reading failed."""
        ok, frame = self.cap.read()
        if not ok:
            return None
        if config.MIRROR_MODE:
            frame = cv2.flip(frame, 1)  # 1 = horizontal flip
        return frame

    def release(self):
        self.cap.release()