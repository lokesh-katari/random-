#!/usr/bin/env python3
"""
3D Spatial Design Canvas — Ursina Engine + MediaPipe Hand Tracking
Excalidraw-style sketchy wireframe aesthetic on a GPU-accelerated 3D canvas.

Features:
  - 60+ FPS GPU-rendered 3D scene via Ursina/Panda3D
  - MediaPipe hand tracking in background thread (zero render blocking)
  - Lerp-smoothed cursor + object movement (anti-jitter)
  - Two-hand independent grab, two-hand zoom/rotate on same object
  - Excalidraw double-stroke wireframe aesthetic

Run:   python3 spatial_canvas.py
Deps:  ursina, opencv-python, mediapipe, numpy
"""

from ursina import *
from ursina.shaders import unlit_shader
import cv2
import numpy as np
import mediapipe as mp_lib
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker, HandLandmarkerOptions, RunningMode
)
import math
import threading
import random
import os
import sys

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")

# Colors (0-1 normalized for Ursina)
BG_COLOR      = color.Color(245/255, 245/255, 245/255, 1)
CHARCOAL      = color.Color(40/255, 40/255, 40/255, 1)
SKETCH_ORANGE = color.Color(255/255, 140/255, 30/255, 1)
SKETCH_BLUE   = color.Color(50/255, 160/255, 230/255, 1)
SKETCH_GREEN  = color.Color(80/255, 200/255, 80/255, 1)
CURSOR_COLORS = [
    color.Color(50/255, 160/255, 230/255, 1),
    color.Color(230/255, 100/255, 50/255, 1),
]

PINCH_THRESHOLD = 0.055
LERP_SPEED      = 0.18       # smoothing factor (lower = smoother, higher = snappier)
OBJ_LERP_SPEED  = 0.14       # object movement smoothing
ROUGHNESS       = 0.012      # sketchy offset in world units

# ─────────────────────────────────────────────
# Wireframe Mesh Generators
# ─────────────────────────────────────────────

def make_box_wireframe(w=1, h=1, d=1):
    """Create a wireframe box mesh (mode='line')."""
    hw, hh, hd = w/2, h/2, d/2
    verts = [
        Vec3(-hw, -hh, -hd), Vec3( hw, -hh, -hd), Vec3( hw,  hh, -hd), Vec3(-hw,  hh, -hd),
        Vec3(-hw, -hh,  hd), Vec3( hw, -hh,  hd), Vec3( hw,  hh,  hd), Vec3(-hw,  hh,  hd),
    ]
    # Line segments: pairs of indices
    edges = [
        0,1, 1,2, 2,3, 3,0,   # front
        4,5, 5,6, 6,7, 7,4,   # back
        0,4, 1,5, 2,6, 3,7,   # connecting
    ]
    return Mesh(vertices=verts, triangles=edges, mode='line', thickness=2)


def make_cylinder_wireframe(radius=0.5, height=1, segments=12):
    """Create a wireframe cylinder mesh (top/bottom rings + verticals)."""
    verts = []
    hh = height / 2
    for ring_y in [-hh, hh]:
        for i in range(segments):
            angle = 2 * math.pi * i / segments
            verts.append(Vec3(radius * math.cos(angle), ring_y, radius * math.sin(angle)))
    edges = []
    n = segments
    for i in range(n):
        # Top ring
        edges.extend([i, (i + 1) % n])
        # Bottom ring
        edges.extend([n + i, n + (i + 1) % n])
        # Verticals
        edges.extend([i, n + i])
    return Mesh(vertices=verts, triangles=edges, mode='line', thickness=2)


def make_cloud_wireframe():
    """Create a warped hexahedron with cross-braces for a cloud look."""
    s = 0.5
    verts = [
        Vec3(-s,     -s*0.6, -s*0.7), Vec3( s,     -s*0.6, -s*0.7),
        Vec3( s*1.1,  s*0.5, -s*0.5), Vec3(-s*0.9,  s*0.6, -s*0.6),
        Vec3(-s*1.1, -s*0.5,  s*0.6), Vec3( s*0.9, -s*0.7,  s*0.7),
        Vec3( s*0.8,  s*0.7,  s*0.5), Vec3(-s*1.0,  s*0.5,  s*0.8),
    ]
    edges = [
        0,1, 1,2, 2,3, 3,0,
        4,5, 5,6, 6,7, 7,4,
        0,4, 1,5, 2,6, 3,7,
        0,6, 1,7, 2,4, 3,5,   # cross-braces
    ]
    return Mesh(vertices=verts, triangles=edges, mode='line', thickness=2)


# ─────────────────────────────────────────────
# SketchAsset — Excalidraw-style 3D Entity
# ─────────────────────────────────────────────

class SketchAsset(Entity):
    """
    A 3D wireframe object with Excalidraw double-stroke aesthetic.
    Two overlapping wireframe copies with micro random offsets simulate the hand-drawn look.
    """

    def __init__(self, name_tag, mesh_fn, position=(0, 0, 0), scale_val=1, **kwargs):
        super().__init__(
            model=mesh_fn(),
            color=CHARCOAL,
            position=position,
            scale=scale_val,
            shader=unlit_shader,
            **kwargs
        )
        self.name_tag = name_tag
        self.unlit = True

        # Second stroke: duplicate mesh with micro offset for sketchy look
        self.stroke2 = Entity(
            parent=self,
            model=mesh_fn(),
            color=CHARCOAL,
            position=Vec3(
                random.uniform(-ROUGHNESS, ROUGHNESS),
                random.uniform(-ROUGHNESS, ROUGHNESS),
                random.uniform(-ROUGHNESS, ROUGHNESS)
            ),
            shader=unlit_shader,
        )

        # Visual state
        self._current_color = CHARCOAL
        self._target_color = CHARCOAL

        # Interaction targets (lerped)
        self.target_pos = Vec3(*position)
        self.target_rot_y = 0.0

        # Label
        self.label = Text(
            text=name_tag,
            scale=0.7,
            color=color.Color(0.5, 0.5, 0.5, 1),
            parent=self,
            y=-0.7,
            billboard=True,
            origin=(0, 0),
        )

    def set_sketch_color(self, col):
        """Set target color (will lerp to it)."""
        self._target_color = col

    def update_visuals(self):
        """Lerp color toward target each frame."""
        t = 0.15
        r = lerp(self.color.r, self._target_color.r, t)
        g = lerp(self.color.g, self._target_color.g, t)
        b = lerp(self.color.b, self._target_color.b, t)
        c = color.Color(r, g, b, 1)
        self.color = c
        self.stroke2.color = c
        self.label.color = c

        # Jitter the second stroke slightly each frame for organic feel
        self.stroke2.position = Vec3(
            random.uniform(-ROUGHNESS, ROUGHNESS),
            random.uniform(-ROUGHNESS, ROUGHNESS),
            random.uniform(-ROUGHNESS, ROUGHNESS)
        )

    def lerp_transform(self, dt):
        """Smoothly interpolate position and rotation toward targets."""
        spd = OBJ_LERP_SPEED
        self.x = lerp(self.x, self.target_pos.x, spd)
        self.y = lerp(self.y, self.target_pos.y, spd)
        self.z = lerp(self.z, self.target_pos.z, spd)
        self.rotation_y = lerp(self.rotation_y, self.target_rot_y, spd)


# ─────────────────────────────────────────────
# Hand Tracking Thread
# ─────────────────────────────────────────────

class HandTracker:
    """
    Runs MediaPipe hand detection in a background thread.
    Produces smoothed landmark data consumed by the main Ursina loop.
    """

    def __init__(self, max_hands=2):
        self.max_hands = max_hands
        self.running = False
        self.thread = None
        self.lock = threading.Lock()

        # Output: list of hand landmark lists (each is 21 landmarks with .x, .y, .z)
        self.raw_landmarks = []      # latest raw from MediaPipe
        self.cap = None

    def start(self):
        """Start the capture + detection thread."""
        if not os.path.isfile(MODEL_PATH):
            print(f"[ERROR] Model not found: {MODEL_PATH}")
            print("  wget -O hand_landmarker.task "
                  "https://storage.googleapis.com/mediapipe-models/"
                  "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task")
            sys.exit(1)

        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        if self.cap:
            self.cap.release()

    def _open_camera(self):
        """Find a working camera (skip virtual devices)."""
        for idx in range(1, 6):
            cap = cv2.VideoCapture(idx)
            if not cap.isOpened():
                continue
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            ok = False
            for _ in range(20):
                ret, _ = cap.read()
                if ret:
                    ok = True
                    break
            if ok:
                print(f"[INFO] Camera index {idx} working.")
                return cap
            cap.release()
        # Fallback: try index 0
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            for _ in range(20):
                ret, _ = cap.read()
                if ret:
                    print("[INFO] Camera index 0 working.")
                    return cap
            cap.release()
        return None

    def _run(self):
        """Background thread: capture frames, detect hands, publish landmarks."""
        self.cap = self._open_camera()
        if self.cap is None:
            print("[WARN] No webcam found. Hand tracking disabled.")
            return

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=RunningMode.VIDEO,
            num_hands=self.max_hands,
            min_hand_detection_confidence=0.65,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.55,
        )
        landmarker = HandLandmarker.create_from_options(options)
        frame_ts = 0

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)  # mirror
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp_lib.Image(image_format=mp_lib.ImageFormat.SRGB, data=rgb)
            frame_ts += 33

            try:
                results = landmarker.detect_for_video(mp_image, frame_ts)
                with self.lock:
                    if results.hand_landmarks:
                        self.raw_landmarks = results.hand_landmarks
                    else:
                        self.raw_landmarks = []
            except Exception:
                pass

        landmarker.close()
        self.cap.release()

    def get_landmarks(self):
        """Thread-safe read of latest landmarks."""
        with self.lock:
            return list(self.raw_landmarks)


# ─────────────────────────────────────────────
# Per-Hand Gesture State
# ─────────────────────────────────────────────

class HandState:
    def __init__(self):
        # Raw input
        self.raw_x = 0.5
        self.raw_y = 0.5
        # Smoothed cursor (in Ursina world-ish coords)
        self.cursor_x = 0.0
        self.cursor_y = 0.0
        self.pinching = False
        self.spin_mode = False
        self.grabbed_obj = None
        self.grab_offset_x = 0.0
        self.grab_offset_y = 0.0
        self.active = False  # whether this hand is being tracked


# ─────────────────────────────────────────────
# Main Application
# ─────────────────────────────────────────────

app = Ursina(
    title='Spatial Canvas — Excalidraw 3D',
    borderless=False,
    fullscreen=False,
    size=(1280, 720),
    development_mode=False,
)

window.color = BG_COLOR
window.fps_counter.enabled = True
camera.orthographic = False
camera.fov = 60
camera.position = Vec3(0, 0, -8)
camera.look_at(Vec3(0, 0, 0))

# ── Dot grid background (Excalidraw-style) ──
grid_dots = []
for gx in range(-12, 13):
    for gy in range(-7, 8):
        dot = Entity(
            model='sphere',
            scale=0.02,
            color=color.Color(0.84, 0.84, 0.84, 1),
            position=Vec3(gx * 0.6, gy * 0.6, 5),
            shader=unlit_shader,
        )
        grid_dots.append(dot)

# ── Scene objects ──
objects = [
    SketchAsset("Server",   make_box_wireframe,      position=(-3, 0, 0),  scale_val=1.5),
    SketchAsset("Database", make_cylinder_wireframe,  position=(0, 0, 0),   scale_val=1.3),
    SketchAsset("Cloud",    make_cloud_wireframe,     position=(3, 0, 0),   scale_val=1.8),
]

# Initialize target positions from placed positions
for obj in objects:
    obj.target_pos = Vec3(obj.x, obj.y, obj.z)
    obj.target_rot_y = obj.rotation_y

# ── Cursors (one per hand) ──
cursors = []
for i in range(2):
    c = Entity(
        model='circle',
        scale=0.15,
        color=CURSOR_COLORS[i],
        position=Vec3(0, 0, -2),
        shader=unlit_shader,
        billboard=True,
        always_on_top=True,
    )
    # Inner dot
    inner = Entity(
        parent=c,
        model='circle',
        scale=0.3,
        color=CURSOR_COLORS[i],
        shader=unlit_shader,
    )
    c.inner = inner
    c.enabled = False
    cursors.append(c)

# ── HUD ──
hud_text = Text(
    text="Pinch = Grab & Move  |  Split Index/Middle = Spin  |  ESC = Quit",
    position=(-0.85, 0.48),
    scale=0.8,
    color=color.Color(0.3, 0.3, 0.3, 1),
)
status_text = Text(
    text="",
    position=(-0.85, 0.44),
    scale=0.7,
    color=SKETCH_BLUE,
)

# ── Hand tracker ──
tracker = HandTracker(max_hands=2)
tracker.start()

hand_states = [HandState(), HandState()]

# Two-hand gesture state
two_hand_obj = None
prev_two_dist = None
prev_two_angle = None


# ─────────────────────────────────────────────
# Utility: Map normalized hand coords to Ursina world
# ─────────────────────────────────────────────

def hand_to_world(nx, ny):
    """Convert normalized hand position (0-1) to Ursina world X, Y."""
    # Map hand x (0=left, 1=right) to roughly -6..+6 world units
    wx = (nx - 0.5) * 12.0
    # Map hand y (0=top, 1=bottom) to roughly +4..-4 world units
    wy = (0.5 - ny) * 8.0
    return wx, wy


def find_closest_object(wx, wy, threshold=1.2):
    """Find the object closest to world cursor position within threshold."""
    best = None
    best_dist = threshold
    for obj in objects:
        dx = obj.x - wx
        dy = obj.y - wy
        dist = math.sqrt(dx*dx + dy*dy)
        if dist < best_dist:
            best_dist = dist
            best = obj
    return best


# ─────────────────────────────────────────────
# Main Update Loop
# ─────────────────────────────────────────────

def update():
    global two_hand_obj, prev_two_dist, prev_two_angle

    dt = time.dt
    landmarks_list = tracker.get_landmarks()
    n_hands = len(landmarks_list)

    # ── Process each hand ──
    for i in range(2):
        hs = hand_states[i]
        if i < n_hands:
            lm = landmarks_list[i]
            idx_tip = lm[8]
            thumb_tip = lm[4]
            mid_tip = lm[12]

            # Raw normalized coords (already mirrored by cv2.flip)
            hs.raw_x = idx_tip.x
            hs.raw_y = idx_tip.y
            hs.active = True

            # Target world position
            twx, twy = hand_to_world(idx_tip.x, idx_tip.y)

            # Lerp smooth the cursor
            hs.cursor_x = lerp(hs.cursor_x, twx, LERP_SPEED)
            hs.cursor_y = lerp(hs.cursor_y, twy, LERP_SPEED)

            # Pinch detection
            pinch_dist = math.hypot(idx_tip.x - thumb_tip.x, idx_tip.y - thumb_tip.y)
            was_pinching = hs.pinching
            hs.pinching = pinch_dist < PINCH_THRESHOLD

            # Spin mode
            idx_mid_dist = math.hypot(idx_tip.x - mid_tip.x, idx_tip.y - mid_tip.y)
            hs.spin_mode = hs.pinching and idx_mid_dist > 0.08

            # Grab logic
            if hs.pinching and not was_pinching:
                target = find_closest_object(hs.cursor_x, hs.cursor_y)
                if target:
                    hs.grabbed_obj = target
                    hs.grab_offset_x = target.target_pos.x - hs.cursor_x
                    hs.grab_offset_y = target.target_pos.y - hs.cursor_y
            elif not hs.pinching:
                hs.grabbed_obj = None

            # Update cursor entity
            cursors[i].enabled = True
            cursors[i].position = Vec3(hs.cursor_x, hs.cursor_y, -2)
            ring_col = CURSOR_COLORS[i] if hs.pinching else color.Color(0.7, 0.7, 0.7, 0.8)
            cursors[i].color = ring_col
            cursors[i].inner.color = ring_col

        else:
            hs.active = False
            hs.pinching = False
            hs.grabbed_obj = None
            cursors[i].enabled = False

    # ── Two-hand same-object gesture ──
    h0, h1 = hand_states
    if (h0.grabbed_obj is not None and h1.grabbed_obj is not None
            and h0.grabbed_obj is h1.grabbed_obj):
        obj = h0.grabbed_obj
        two_hand_obj = obj

        dx = h1.cursor_x - h0.cursor_x
        dy = h1.cursor_y - h0.cursor_y
        dist = math.hypot(dx, dy)
        ang = math.atan2(dy, dx)

        if prev_two_dist is not None and dist > 0.1:
            # Scale
            scale_delta = (dist - prev_two_dist) * 0.3
            new_scale = max(0.4, min(4.0, obj.scale_x + scale_delta))
            obj.scale = Vec3(new_scale, new_scale, new_scale)

            # Rotate
            angle_delta = ang - prev_two_angle
            if angle_delta > math.pi:
                angle_delta -= 2 * math.pi
            elif angle_delta < -math.pi:
                angle_delta += 2 * math.pi
            obj.target_rot_y += math.degrees(angle_delta)

            # Move (midpoint)
            mid_x = (h0.cursor_x + h1.cursor_x) / 2
            mid_y = (h0.cursor_y + h1.cursor_y) / 2
            obj.target_pos = Vec3(mid_x, mid_y, obj.target_pos.z)

        prev_two_dist = dist
        prev_two_angle = ang

    else:
        two_hand_obj = None
        prev_two_dist = None
        prev_two_angle = None

        # Single-hand gestures
        for hs in hand_states:
            if hs.grabbed_obj is not None and hs.pinching:
                obj = hs.grabbed_obj
                if hs.spin_mode:
                    # Horizontal hand movement → Y rotation
                    twx, _ = hand_to_world(hs.raw_x, hs.raw_y)
                    rot_delta = (twx - hs.cursor_x) * 15.0
                    obj.target_rot_y += rot_delta
                else:
                    obj.target_pos = Vec3(
                        hs.cursor_x + hs.grab_offset_x,
                        hs.cursor_y + hs.grab_offset_y,
                        obj.target_pos.z
                    )

    # ── Update object visuals ──
    for obj in objects:
        # Determine color
        if two_hand_obj is obj:
            obj.set_sketch_color(SKETCH_GREEN)
        elif any(hs.grabbed_obj is obj for hs in hand_states):
            obj.set_sketch_color(SKETCH_BLUE)
        elif any(
            hs.active and not hs.pinching
            and abs(hs.cursor_x - obj.x) < 1.2
            and abs(hs.cursor_y - obj.y) < 1.2
            for hs in hand_states
        ):
            obj.set_sketch_color(SKETCH_ORANGE)
        else:
            obj.set_sketch_color(CHARCOAL)

        obj.update_visuals()
        obj.lerp_transform(dt)

    # ── HUD ──
    parts = []
    for i, hs in enumerate(hand_states):
        if hs.grabbed_obj:
            mode = "SPIN" if hs.spin_mode else "MOVE"
            parts.append(f"H{i+1}: {hs.grabbed_obj.name_tag} [{mode}]")
    if two_hand_obj:
        parts.append(f"2-HAND: {two_hand_obj.name_tag} [ZOOM/ROTATE] s={two_hand_obj.scale_x:.2f}")
    status_text.text = "  |  ".join(parts) if parts else ("No hands detected" if n_hands == 0 else "")


def input(key):
    if key == 'escape':
        tracker.stop()
        application.quit()


app.run()
