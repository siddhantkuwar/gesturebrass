import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)
hands = mp.solutions.hands.Hands()
draw = mp.solutions.drawing_utils

while True:
    ret, f = cap.read()
    if not ret: break

    res = hands.process(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    if res.multi_hand_landmarks:
        for hl in res.multi_hand_landmarks:
            draw.draw_landmarks(f, hl, mp.solutions.hands.HAND_CONNECTIONS)

    cv2.imshow("TEST", f)
    if cv2.waitKey(1)&0xFF==27:
        break

cap.release()
cv2.destroyAllWindows()
