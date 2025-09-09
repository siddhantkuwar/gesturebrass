import cv2
import mediapipe as mp
import math
import numpy as np
import simpleaudio as sa
import time

# ─── PARAMETERS ─────────────────────────────────────────────────────────────
SAMPLE_RATE    = 44100
DURATION       = 5.0     # seconds buffer for holding a note
CURL_THRESH    = 0.1     # fingertip–PIP threshold for valve press
PINCH_THRESH   = 0.04    # thumb-index distance to trigger volume mode
VOL_MIN_DIST   = PINCH_THRESH
VOL_MAX_DIST   = 0.15    # thumb-index max separation for full volume

# MediaPipe fingertip & PIP joint indices for valves
TIP_IDS = [8, 12, 16]   # index, middle, ring tips
PIP_IDS = [6, 10, 14]   # corresponding PIP joints

# ─── PITCH MAPPING ─────────────────────────────────────────────────────────
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
NOTE_NAME_MAP = {combo: name for combo, name in zip(VALVE_NOTE_MAP.keys(),
    ['C4','A3','B3','G#3','A#3','G3','F#3','F3'])}
NOTE_COLOR_MAP = {
    'C4': (255,255,255), 'A3': (0,255,0), 'B3': (0,0,255),
    'G#3': (255,0,0), 'A#3': (0,255,255), 'G3': (255,0,255),
    'F#3': (255,165,0), 'F3': (128,0,128)
}

# ─── UTILS ──────────────────────────────────────────────────────────────────
def dist(a, b):
    """Euclidean distance between two normalized landmarks."""
    return math.hypot(a.x - b.x, a.y - b.y)


def make_sine(freq, amp):
    """Generate int16 numpy buffer for a sine wave."""
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), False)
    wave = np.sin(freq * t * 2 * np.pi)
    audio = wave * amp * (2**15 - 1)
    return audio.astype(np.int16)

# ─── MAIN ───────────────────────────────────────────────────────────────────
def main():
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
    prev_time    = time.time()
    vol_mode     = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # calculate FPS
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if curr_time != prev_time else 0
        prev_time = curr_time

        # prepare image
        img_rgb = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        img_rgb.flags.writeable = False
        results = hands.process(img_rgb)
        img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        h, w, _ = img.shape

        valve_states = [0, 0, 0]
        velocity     = 0.0
        show_vol     = False
        vol_pos      = (0, 0)
        left_seen    = False

        if results.multi_hand_landmarks:
            for hl, handedness in zip(results.multi_hand_landmarks,
                                      results.multi_handedness):
                label = handedness.classification[0].label
                # get pixel coords and distances
                d_idx = dist(hl.landmark[4], hl.landmark[8])
                ix_pix = int(hl.landmark[8].x * w)
                iy_pix = int(hl.landmark[8].y * h)
                tx_pix = int(hl.landmark[4].x * w)
                ty_pix = int(hl.landmark[4].y * h)

                if label == 'Left':
                    left_seen = True
                    # enter vol mode on pinch
                    if d_idx < PINCH_THRESH:
                        vol_mode = True
                    # if in vol mode, update volume
                    if vol_mode:
                        show_vol = True
                        norm = (d_idx - VOL_MIN_DIST) / (VOL_MAX_DIST - VOL_MIN_DIST)
                        velocity = max(0, min(norm, 1))
                        # set display pos
                        cx = (ix_pix + tx_pix) // 2
                        cy = (iy_pix + ty_pix) // 2
                        vol_pos = (cx + 20, cy - 50)
                else:
                    # right hand valves
                    for i, (tip, pip) in enumerate(zip(TIP_IDS, PIP_IDS)):
                        d = dist(hl.landmark[tip], hl.landmark[pip])
                        valve_states[i] = 1 if d < CURL_THRESH else 0

                draw_utils.draw_landmarks(img, hl, hands_module.HAND_CONNECTIONS)

        # reset vol_mode if left hand left screen
        if not left_seen:
            vol_mode = False

        # map pitch & play audio on valve change
        combo = tuple(valve_states)
        freq  = VALVE_NOTE_MAP.get(combo, 261.63)
        note_name = NOTE_NAME_MAP.get(combo, 'C4')
        color     = NOTE_COLOR_MAP.get(note_name, (255,255,255))

        if valve_states != prev_valves:
            amp = velocity or 0.5
            if any(valve_states):
                if play_obj:
                    play_obj.stop()
                buffer = make_sine(freq, amp)
                play_obj = sa.play_buffer(buffer, 1, 2, SAMPLE_RATE)
            else:
                if play_obj:
                    play_obj.stop()
                    play_obj = None
            prev_valves = valve_states.copy()

        # overlays: FPS & note
        cv2.putText(img, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (200,200,0), 2)
        cv2.putText(img, f"Note: {note_name}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # draw volume bar if active
        if show_vol:
            bx, by = vol_pos
            bar_w, bar_h = 30, 200
            cv2.rectangle(img, (bx, by), (bx+bar_w, by+bar_h), (50,50,50), -1)
            fh = int(bar_h * velocity)
            cv2.rectangle(img, (bx, by+bar_h-fh), (bx+bar_w, by+bar_h), (100,200,100), -1)
            cv2.putText(img, f"{int(velocity*100)}%", (bx+bar_w+5, by+bar_h//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        cv2.imshow('GestureBrass', img)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
