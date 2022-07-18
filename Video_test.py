from sre_constants import SUCCESS
import numpy as np
from Libs.HandTracker import HandTracker
import cv2
import pickle
from Libs.utils import camera

def distanceMatrix(handData, palmdist):
    distMatrix = np.zeros([len(handData), len(handData)], dtype="float32")
    pd = palmdist[0]
    for i in range(len(handData)):
        if i > 20:
            pd = palmdist[1]
        for j in range(len(handData)):
            distMatrix[i][j] = np.linalg.norm(np.array(handData[i]) - np.array(handData[j])) /pd
    return distMatrix.reshape(21*21)


def getData(image):
    tracker = HandTracker()
    results = tracker.findHands(image)
    positions = []
    palmdist = []
    if results:
        for landmarks in results:
            tracker.drawHands(image, landmarks)
            position = tracker.getPos(landmarks.landmark, image.shape)
            pd = np.linalg.norm(np.array(position[0]) - np.array(position[9]))
            palmdist.append(pd)
            positions.extend(position)
        return distanceMatrix(positions, palmdist)
    else:
        return []

with open('digits_classifier2.pkl', 'rb') as f:
    model = pickle.load(f)

cap = camera(640,480,0)

while True:
    success,image = cap.read()
    image = cv2.flip(image,1)
    test = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    if success:
        data = getData(test)
        if len(data):
            # print(data.ndim)
            print(model.predict([data]))
        cv2.imshow("Output", image)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break

