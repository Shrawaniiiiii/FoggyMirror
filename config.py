"""Central configuration for the Foggy Mirror project.

Every tunable number lives here so you can experiment
without touching the logic files.
"""

# ---------- Camera ----------
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
TARGET_FPS = 30
USE_MJPG = True
MIRROR_MODE = True

# ---------- Window ----------
WINDOW_NAME = "Foggy Mirror"
SHOW_FPS = True
SHOW_INSTRUCTIONS = True
SHOW_DEBUG = False        # tracking overlay (toggle with D)

# ---------- Face / mouth tracking ----------
TRACK_WIDTH = 480
FACE_EVERY_N_FRAMES = 2        # 1 = every frame (more accurate, slower)
MOUTH_RATIO_SMOOTHING = 0.5
MOUTH_OPEN_ON = 0.30
MOUTH_OPEN_OFF = 0.20

# ---------- Hand tracking ----------
HAND_TRACK_WIDTH = 320         # frame width given to MediaPipe Hands
HAND_MODEL_COMPLEXITY = 0      # 0 = fast, 1 = more accurate
HAND_DETECTION_CONF = 0.6
HAND_TRACKING_CONF = 0.5
EXTEND_RATIO = 1.10            # tip/joint distance ratio that counts as "extended"
GESTURE_HOLD_FRAMES = 2        # frames a new gesture must persist to be accepted
ONE_EURO_MIN_CUTOFF = 2.0      # lower = smoother when slow
ONE_EURO_BETA = 12.0           # higher = less lag when fast

# ---------- Finger writing ----------
FINGER_RADIUS_RATIO = 0.11     # finger radius = hand size * this
FINGER_RADIUS_MIN = 0.012      # (all radii are fractions of frame height)
FINGER_RADIUS_MAX = 0.045
FINGER_STRENGTH = 1.0

# ---------- Palm wiping ----------
PALM_RADIUS_RATIO = 0.55
PALM_RADIUS_MIN = 0.06
PALM_RADIUS_MAX = 0.20
PALM_STAMP_STRENGTH = 0.30     # per stamp; stamps accumulate within a swipe
PALM_MIN_SPEED = 0.35          # frame-heights/second needed before wiping starts

# ---------- Stroke behaviour ----------
BRUSH_SPACING = 0.30           # distance between stamps (fraction of radius)
STROKE_JUMP_LIMIT = 0.5        # tracking jump (in frame heights) that breaks a stroke
STROKE_LOST_GRACE = 3          # frames a stroke survives a tracking dropout

# ---------- Fog simulation ----------
FOG_SCALE = 0.5
FOG_DENSITY = 0.92
FOG_FORM_RATE = 0.35
FOG_FADE_RATE = 0.015
FOG_SPREAD = 0.25
BREATH_SIGMA = 0.32
BREATH_RISE_SPEED = 2.5
BREATH_FALL_SPEED = 1.5

# ---------- Realism ----------
FOG_REGROW_RATE = 0.012        # how fast wiped areas fog over again (per second)
REFOG_BY_BREATH = 0.25         # breathing on wiped glass re-fogs it
RIM_STRENGTH = 0.6             # detection strength of stroke edges
RIM_ALPHA = 0.20               # extra condensation along stroke edges
RIM_BRIGHT = 0.07              # brightness of the moisture rim
RESIDUAL_DROPLETS = 0.35       # tiny beads left on wiped glass
DRIPS_ENABLED = False
DRIP_SPAWN_RATE = 0.6          # new drips per second
DRIP_MAX = 14
VIGNETTE = 0.16                # fog slightly denser toward the edges

# ---------- Fog look ----------
FOG_TINT_BGR = (230, 222, 214)
FOG_BRIGHTNESS_FLOOR = 0.55
FOG_BLUR_SIGMA = 3.0