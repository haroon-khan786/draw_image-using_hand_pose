import mediapipe as mp
import cv2

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils



while True:
    ret , frame = cap.read()


    frameRGB = cv2.cvtColor(frame , cv2.COLOR_BGR2RGB)

    results = hands.process(frameRGB)
    print(results.multi_hand_landmarks) 

    if results.multi_hand_landmarks:
        for handlms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(frame , handlms)

    cv2.imshow('frame' , frame)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()