from cv2 import cv2
import os
import HandTrackingModule as htm

wCam, hCam = 1080, 720

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
folderPath = 'Fingers'
myList = os.listdir(folderPath)
overLayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overLayList.append(image)

detector = htm.handDetector(detectionCon=0.75)
tipsId = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()
    img = detector.findHand(img)
    lmList = detector.findPosition(img,handNo=0, draw=False)

    if len(lmList) != 0:
        fingers = []
        # Thumb
        if lmList[tipsId[0]][1] > lmList[tipsId[0] - 1][1]:

            fingers.append(1)
        else:
            fingers.append(0)

        # 4 Fingers
        for id in range(1, 5):
            if lmList[tipsId[id]][2] < lmList[tipsId[id]-2][2]:

                fingers.append(1)
            else:
                fingers.append(0)
        totalFingers = fingers.count(1)
        print(totalFingers)

        h, w, c = overLayList[totalFingers-1].shape
        img[0:h, 0:w] = overLayList[totalFingers-1]

        cv2.rectangle(img, (45,325), (178, 425), (255,255,0), cv2.FILLED)
        cv2.putText(img, str(totalFingers), (95, 400), cv2.FONT_HERSHEY_PLAIN, 4, (255, 0, 0), 10)


    fps = cap.get(cv2.CAP_PROP_FPS)
    cv2.putText(img, f'FPS: {int(fps)}', (680, 70), cv2.FONT_HERSHEY_PLAIN,
                                                         2, (255, 0, 0), 2)

    cv2.imshow('image', img)
    cv2.waitKey(1)