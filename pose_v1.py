import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Start video capture
cap = cv2.VideoCapture(0)


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame for better user experience
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe
    result = hands.process(rgb_frame)

    # Drawing logic
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            
            # Extract index finger tip (point 8)
            index_finger_tip = hand_landmarks.landmark[8]
            
            h , w, _ = frame.shape
            start_x , start_y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
                
                
            # Access specific landmarks
            landmarks = hand_landmarks.landmark

            INDEX_FINGER_TIP = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            THUMB_TIP = landmarks[mp_hands.HandLandmark.THUMB_TIP]
            if THUMB_TIP.y > INDEX_FINGER_TIP.y:
                # If the drawing frame is initialized, start drawing
                cv2.circle(frame, (start_x, start_y), 10, (0, 0, 255), -1)

     
    cv2.imshow("Finger Drawing", frame)

    # Exit condition
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
