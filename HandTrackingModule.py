from cv2 import cv2
import mediapipe as mp
import time

class handDetector():

    def __init__(self,mode=False, maxHands = 2, model_complexity = 1, detectionCon=0.5, trackCon=0.5):

        self.mode = mode
        self.maxHands = maxHands
        self.model_complexity = model_complexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHand = mp.solutions.hands
        self.hands = self.mpHand.Hands(self.mode, self.maxHands, self.model_complexity, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHand(self, img, draw=True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgRGB_flip = cv2.flip(imgRGB,1)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHand.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo:0, draw=True):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 7, (0, 0, 255), cv2.FILLED)
        return lmList


def main():

    cTime = 0
    pTime = 0
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    detector = handDetector()

    while True:

        success, img = cap.read()
        img = detector.findHand(img)
        lmList = detector.findPosition(img, handNo=0)
        if len(lmList) !=0:
            print(lmList[4])
        cTime = time.time()
        fps = 1 /(cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow('image', img)
        cv2.waitKey(10)

if __name__ == "__main__":
    main()