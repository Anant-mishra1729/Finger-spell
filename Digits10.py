from sre_constants import SUCCESS
import numpy as np
from Libs.HandTracker import HandTracker
import cv2
import pickle
from Libs.utils import camera,Fps
from Libs.Cam import WebcamVideoStream
from Libs.FPS import FPS
def distanceMatrix(handData, palmdist):
    """Eucledian distances between points 0,4,5,9,13,17,8,12,16 and 20"""
    distMatrix = np.zeros([len(handData), len(handData)], dtype="float32")
    pd = palmdist[0]
    for i in range(len(handData)):
        if i > 9:
            pd = palmdist[1]
        for j in range(len(handData)):
            distMatrix[i][j] = np.linalg.norm(np.array(handData[i]) - np.array(handData[j])) /pd
    return distMatrix


with open('Digits10_classifier.pkl', 'rb') as f:
    model = pickle.load(f)

cap = WebcamVideoStream(src = 0).start()
tracker = HandTracker()
points = [0, 4, 5, 9, 13, 17, 8, 12, 16, 20]
fps = FPS().start()

while True:
    image = cap.read()
    image = cv2.flip(image,1)
    Fps(image)
    results = tracker.findHands(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
    if results:
        positions = []
        palmdist = []
        for landmarks in results:
            tracker.drawHands(image,landmarks)
            position = tracker.getPos(landmarks.landmark, image.shape,points=points)
            pd = np.linalg.norm(np.array(position[0]) - np.array(position[3]))
            palmdist.append(pd)
            positions.extend(position)
            data = distanceMatrix(positions, palmdist)
            if len(data) == 10:
                count = 0
                print(model.predict([data.reshape(10*10)]))
    cv2.imshow("Output", image)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
    fps.update()
fps.stop()
cv2.destroyAllWindows()
cap.stop()
