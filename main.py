"""Foggy Mirror - final version: breath fog, finger writing, palm wiping."""

import time

import cv2
import numpy as np

import config
from camera import Camera
from face_tracker import FaceTracker
from fog_engine import FogEngine
from hand_tracker import HandTracker, HAND_CONNECTIONS
from interaction import Interaction
from renderer import FogRenderer


def draw_text(img, text, org, scale=0.55, color=(255, 255, 255)):
    cv2.putText(img, text, (org[0] + 1, org[1] + 1), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, 1, cv2.LINE_AA)


def draw_face_debug(img, face):
    h, w = img.shape[:2]
    color = (0, 220, 0) if face.is_open else (60, 60, 255)
    for (x, y) in face.mouth_points:
        cv2.circle(img, (int(x * w), int(y * h)), 3, color, -1, cv2.LINE_AA)

    bx, by, bw, bh = 15, 55, 200, 12
    cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (255, 255, 255), 1)
    fill = int(bw * float(np.clip(face.ratio / 0.6, 0.0, 1.0)))
    cv2.rectangle(img, (bx, by), (bx + fill, by + bh), color, -1)
    for t in (config.MOUTH_OPEN_OFF, config.MOUTH_OPEN_ON):
        tx = bx + int(bw * t / 0.6)
        cv2.line(img, (tx, by - 3), (tx, by + bh + 3), (255, 255, 0), 1)

    status = "OPEN" if face.is_open else "closed"
    face_txt = "face found" if face.found else "NO FACE"
    draw_text(img, f"{face_txt} | ratio {face.ratio:.2f} | mouth {status}",
              (15, 90), 0.5, color)


def draw_hand_debug(img, hand):
    h, w = img.shape[:2]
    if not hand.found:
        draw_text(img, "NO HAND", (15, 115), 0.5, (60, 60, 255))
        return
    pts = [(int(x * w), int(y * h)) for x, y in hand.points]
    for a, b in HAND_CONNECTIONS:
        cv2.line(img, pts[a], pts[b], (0, 200, 255), 1, cv2.LINE_AA)
    for p in pts:
        cv2.circle(img, p, 2, (0, 200, 255), -1, cv2.LINE_AA)
    if hand.gesture == "index":
        cv2.circle(img, pts[8], 7, (0, 255, 0), 1, cv2.LINE_AA)
    draw_text(img, f"gesture: {hand.gesture} | speed {hand.speed:.2f} | "
                   f"size {hand.hand_size:.2f}", (15, 115), 0.5, (0, 200, 255))


def main():
    camera = Camera()
    print(f"Camera opened at {camera.width}x{camera.height}")
    print("Keys: D debug | F fog on/off | C clear | G fog all | S screenshot | "
          "[ ] density | Q/ESC quit")

    first = camera.read()
    if first is None:
        raise RuntimeError("Camera opened but no frame could be read.")
    fh, fw = first.shape[:2]

    face_tracker = FaceTracker()
    hand_tracker = HandTracker()
    fog = FogEngine(fw, fh)
    renderer = FogRenderer(fw, fh)
    interaction = Interaction(fog, fw, fh)

    cv2.namedWindow(config.WINDOW_NAME, cv2.WINDOW_NORMAL)

    show_fog = True
    show_debug = config.SHOW_DEBUG
    save_requested = False
    fps = 0.0
    prev = time.perf_counter()

    try:
        while True:
            frame = camera.read()
            if frame is None:
                print("Failed to read a frame from the camera.")
                break

            now = time.perf_counter()
            dt = min(now - prev, 0.1)
            prev = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

            # --- tracking ---
            face = face_tracker.process(frame)
            hand = hand_tracker.process(frame, now)

            # --- simulation ---
            fog.update(dt, face.is_open, face.mouth_center)
            interaction.update(hand)

            # --- rendering ---
            if show_fog:
                out = renderer.render(frame, fog, fog.density)
            else:
                out = frame.copy()

            if save_requested:
                name = f"foggy_mirror_{int(time.time())}.png"
                cv2.imwrite(name, out)
                print(f"Saved {name}")
                save_requested = False

            if show_debug:
                draw_face_debug(out, face)
                draw_hand_debug(out, hand)
            if config.SHOW_FPS:
                draw_text(out, f"FPS: {fps:.0f}   fog: {fog.mask.mean():.2f}   "
                               f"density: {fog.density:.2f}", (15, 30))
            if config.SHOW_INSTRUCTIONS:
                draw_text(out, "Mouth: fog | Index finger: write | Open palm + move: wipe | "
                               "D debug | S save | Q quit",
                          (15, out.shape[0] - 15), 0.42)

            cv2.imshow(config.WINDOW_NAME, out)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("f"):
                show_fog = not show_fog
            elif key == ord("d"):
                show_debug = not show_debug
            elif key == ord("c"):
                fog.clear()
            elif key == ord("g"):
                fog.fog_all()
            elif key == ord("s"):
                save_requested = True
            elif key == ord("["):
                fog.adjust_density(-0.05)
            elif key == ord("]"):
                fog.adjust_density(0.05)

            if cv2.getWindowProperty(config.WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        face_tracker.close()
        hand_tracker.close()
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()