from os import walk,path
import cv2

def getFiles(filePath,exts):
    imgPaths = []
    for (root, dirs, files) in walk(filePath):
        for f in files:
            if path.splitext(f)[1] in exts:
                imgPaths.append(path.join(root, f))
    return imgPaths

def camera(width,height):
    cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,height)
    cap.set(cv2.CAP_PROP_FPS,30)
    cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'))
    return cap

def main():
	imgPath = ""
	print(getFiles(imgPath,(".jpg")))

if __name__ == '__main__':
	main()