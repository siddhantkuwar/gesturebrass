import cv2
import mediapipe as mp
import math
import numpy as np
import simpleaudio as sa

# ─── PARAMETERS ─────────────────────────────────────────────────────────────
SAMPLE_RATE = 44100
DURATION    = 5.0     # seconds buffer for holding a note
CURL_THRESH = 0.1     # tweak if needed
VOLUME_MIN  = 0.02
VOLUME_MAX  = 0.17

# MediaPipe fingertip & PIP joint indices for valves
TIP_IDS = [8, 12, 16]   # index, middle, ring
PIP_IDS = [6, 10, 14]

# ─── PITCH MAPPING ─────────────────────────────────────────────────────────
# Map 3-valve combinations to frequencies (Hz)
VALVE_NOTE_MAP = {
    (0, 0, 0): 261.63,  # C4
    (1, 0, 0): 220.00,  # A3
    (0, 1, 0): 246.94,  # B3
    (1, 1, 0): 207.65,  # G#3
    (0, 0, 1): 233.08,  # A#3
    (1, 0, 1): 196.00,  # G3
    (0, 1, 1): 184.99,  # F#3
    (1, 1, 1): 174.61,  # F3
}


def dist(a, b):
    """Euclidean distance between two normalized landmarks."""
    return math.hypot(a.x - b.x, a.y - b.y)


def make_sine(freq, amp):
    """Generate int16 numpy buffer for a sine wave."""
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), False)
    wave = np.sin(freq * t * 2 * np.pi)
    audio = wave * amp * (2**15 - 1)
    return audio.astype(np.int16)


def main():
    # MediaPipe Hands setup
    hands_module = mp.solutions.hands
    hands = hands_module.Hands(
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )
    draw_utils = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam")
        return

    prev_valves = [0, 0, 0]
    play_obj     = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Mirror & convert to RGB for detection
        img = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        img.flags.writeable = False
        results = hands.process(img)

        # Back to BGR for display
        img.flags.writeable = True
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        valve_states = [0, 0, 0]
        velocity     = 0

        if results.multi_hand_landmarks:
            for hl, handedness in zip(results.multi_hand_landmarks,
                                      results.multi_handedness):
                label = handedness.classification[0].label
                if label == 'Right':
                    # Detect finger curls → valves
                    for i, (tip, pip) in enumerate(zip(TIP_IDS, PIP_IDS)):
                        d = dist(hl.landmark[tip], hl.landmark[pip])
                        valve_states[i] = 1 if d < CURL_THRESH else 0
                else:
                    # Left hand → volume control
                    d = dist(hl.landmark[4], hl.landmark[20])
                    norm = min(max((d - VOLUME_MIN) / (VOLUME_MAX - VOLUME_MIN), 0), 1)
                    velocity = norm  # 0.0–1.0

        # Send audio on state change with pitch mapping
        if valve_states != prev_valves:
            combo = tuple(valve_states)
            freq = VALVE_NOTE_MAP.get(combo, 261.63)
            amp  = velocity or 0.5

            if any(valve_states):  # note_on
                if play_obj:
                    play_obj.stop()
                buffer = make_sine(freq, amp)
                play_obj = sa.play_buffer(buffer, 1, 2, SAMPLE_RATE)
            else:  # note_off
                if play_obj:
                    play_obj.stop()
                    play_obj = None

            prev_valves = valve_states.copy()

        # Draw landmarks
        if results.multi_hand_landmarks:
            for hl in results.multi_hand_landmarks:
                draw_utils.draw_landmarks(img, hl, hands_module.HAND_CONNECTIONS)

        cv2.imshow('GestureBrass', img)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
