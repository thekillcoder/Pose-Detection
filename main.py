import pose_detection as pd
import mediapipe as mp
import numpy as np
import cv2 as cv
import time

def main():
    print("Starting webcam...")
    cap = cv.VideoCapture(0)

    detector = pd.PoseDetector()
    print("PoseDetector initialized")

    ptime = 0

    while True:
        success, img = cap.read()
        if not success:
            print("Failed to capture frame")
            break

        print("Frame captured")

        flipped_img = cv.flip(img, 1)
        img = detector.findPose(img, flipped_img, draw=True)
        lmlist = detector.findPosition(img, draw=True)

        cTime = time.time()
        fps = 1 / (cTime - ptime) if cTime != ptime else 0
        ptime = cTime

        cv.putText(img, f'FPS: {int(fps)}', (10, 70), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        cv.imshow("Partner Pose", img)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()