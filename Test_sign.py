from libs.HandTracker import HandTracker
from libs.utils import camera, getFiles, Fps
import cv2
import numpy as np
import pickle

tracker = HandTracker()
cap = camera(640, 320)
files = getFiles("Output", (".pickle"))
points = [0, 4, 5, 9, 13, 17, 8, 12, 16, 20]

filedict = []
for file in files:
    with open(file, 'rb') as handle:
        filedict.append(pickle.load(handle))

print(filedict)

def eucledianDistances(handData, palmdist):
    distMatrix = np.zeros([len(handData), len(handData)], dtype='float')
    pd = palmdist[0]
    for i in range(len(handData)):
        if (i > 9):
            pd = palmdist[1]
        for j in range(len(handData)):
            distMatrix[i][j] = (
                ((handData[i][0]-handData[j][0])**2 + (handData[i][1]-handData[j][1])**2)**0.5)/pd
    return distMatrix


def findError(known, test):
    return int(np.sum(np.absolute(known - test)))


# For testing using image

# test_image = cv2.imread("Resources\\ISL\\Indian\\1\\1.jpg")
# results = tracker.findHands(cv2.cvtColor(test_image,cv2.COLOR_RGB2BGR))
# if results:
# 	positions = []
# 	palmdist = []
# 	for landmarks in results:
# 		position = tracker.getPos(landmarks.landmark,test_image.shape,points)
# 		pd = ((position[0][0]-position[3][0])**2 + (position[0][1]-position[3][1])**2)**0.5
# 		palmdist.append(pd)
# 		positions.extend(position)
# 	test = eucledianDistances(positions,palmdist)
# known = (filedict[0])[0]
# print(findError(known,test))


# For testing using camera


count = 0
while True:
    success, image = cap.read()
    if success:
        image = cv2.flip(image, 1)
        results = tracker.findHands(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        positions = []
        palmdist = []
        if results:
            for landmarks in results:
                # tracker.drawHands(image,landmarks)
                position = tracker.getPos(landmarks.landmark, image.shape, points)
                palmdist.append(((position[0][0]-position[3][0])**2 + (position[0][1]-position[3][1])**2)**0.5)
                positions.extend(position)
            test = eucledianDistances(positions, palmdist)
            err_matrix = []
            for gesture in filedict:
                if len(gesture[0]) == len(test):
                    err_matrix.append((findError(gesture[0], test), gesture[1]))
            if len(err_matrix):
                cv2.putText(image, min(err_matrix)[1], (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        Fps(image)
        cv2.imshow("Output", image)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break
    else:
        print("Camera not found!")
        break
