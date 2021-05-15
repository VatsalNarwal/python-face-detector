import sys
import cv2
from random import randrange

# Load trained data from xml file
td = cv2.CascadeClassifier('trained_data/haarcascade_frontalface_default.xml')

wc = cv2.VideoCapture(0)

color = (randrange(256), randrange(256), randrange(256))

while True:
    successful_frame_read, frame = wc.read()

    # GrayScale the frame
    grayScaleImg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Get face cordinates
    faceCordinates = td.detectMultiScale(grayScaleImg)


    for (x, y, w, h) in faceCordinates:

        print(f"Face Cordinates ➜ {faceCordinates[0]}")

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    # Show up the image
    cv2.imshow("Face Detector", frame)

    cv2.waitKey(1)

# Wait for a key to be pressed
cv2.waitKey()

sys.exit("Process completed!")
