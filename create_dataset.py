from turtle import position
from unittest import result
from Libs.HandTracker import HandTracker
from Libs.utils import camera
import cv2
tracker = HandTracker()
cap = camera(640,480,0)
count = 0
points = [0, 4, 5, 9, 13, 17, 8, 12, 16, 20]

while True:
    _,image = cap.read()
    copied = image.copy()
    results = tracker.findHands(image)
    key = cv2.waitKey(1)
    if results:
        positions = []
        for landmarks in results:
            tracker.drawHands(copied,landmarks)
            position = tracker.getPos(landmarks.landmark, image.shape, points)
            positions.extend(position)
        if len(positions) == 20 and key == ord('s'):
            print(len(positions))
            for count in range(500):
                cv2.imwrite(f"Datasets\\W\\{count}.jpg",image)
            break
    cv2.imshow("Output",copied)
    if key == ord('q'):
        break