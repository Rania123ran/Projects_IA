import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Load video capture
cap = cv2.VideoCapture(0)

# Initialize Mediapipe hands
with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while True:
        # Read frame from camera
        ret, frame = cap.read()

        # Convert image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Flip image horizontally
        image = cv2.flip(image, 1)

        # Set flag to detect landmarks
        results = hands.process(image)

        # Draw landmarks on image
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)     
        
        # Detect finger count
        finger_count = 0
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            tip_ids = [4, 8, 12, 16, 20]  # Landmark ids of finger tips
            finger_states = []
            for tip_id in tip_ids:
                finger_tip = hand_landmarks.landmark[tip_id]
                finger_mcp = hand_landmarks.landmark[tip_id - 1]
                # Check if finger is open or closed
                if tip_id==4:
                    finger_states.append(finger_tip.x < finger_mcp.x)
                else:
                    finger_states.append(finger_tip.y < finger_mcp.y)
            # Count number of open fingers
            finger_count = finger_states.count(True)

        # Display finger count on image
        cv2.putText(image, str(finger_count), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Display image
        cv2.imshow('Finger Counter', image)

        # Check for 'q' key to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()