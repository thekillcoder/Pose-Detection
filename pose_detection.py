import cv2 as cv
import mediapipe as mp

import mediapipe as mp

class PoseDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose          # ✅ Store the pose module
        self.pose = self.mp_pose.Pose()           # ✅ Initialize the Pose object
        self.mp_draw = mp.solutions.drawing_utils
        self.results = None

    def findPose(self, img, flipped_img, draw=True):
         imgRGB = cv.cvtColor(flipped_img, cv.COLOR_BGR2RGB)
         self.results = self.pose.process(imgRGB) 

         if self.results.pose_landmarks and draw:
          self.mp_draw.draw_landmarks(img, self.results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS) 
         return img
    
    def findPosition(self, img, draw=True):
        lmList = []
        if self.results and self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append((id, cx, cy))
                if draw:
                    cv.circle(img, (cx, cy), 5, (0, 0, 255), cv.FILLED)
        return lmList