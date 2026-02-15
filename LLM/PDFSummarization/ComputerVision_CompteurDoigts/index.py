import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)
tip_ids = [4, 8, 12, 16, 20]

with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=1) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        finger_count = 0
        hand_label = "Unknown"

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            lm = hand_landmarks.landmark

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            # Si 5.x < 17.x => main droite (dans l'image)
            hand_label = "Right" if lm[5].x < lm[17].x else "Left"
            finger_states = []
            for tip in tip_ids:
                tip_pt = lm[tip]
                prev_pt = lm[tip - 1]

                if tip == 4:
                    if hand_label == "Right":
                        finger_states.append(tip_pt.x < prev_pt.x)
                    else:
                        finger_states.append(tip_pt.x > prev_pt.x)
                else:
                    finger_states.append(tip_pt.y < prev_pt.y)

            finger_count = finger_states.count(True)

            cv2.putText(frame, f"Hand: {hand_label}", (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.rectangle(frame, (10, 10), (260, 60), (0, 0, 0), -1)
        cv2.putText(frame, f"Fingers: {finger_count}", (20, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        cv2.imshow("Finger Counter", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
