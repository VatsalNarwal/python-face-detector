import sys
import cv2
from random import randrange

# Load trained data from xml file
td = cv2.CascadeClassifier('trained_data/haarcascade_frontalface_default.xml')

# Read image from file
img = cv2.resize(cv2.imread('_data/em.png'), (600, 540))

# GrayScale the image
grayScaleImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Get face cordinates
faceCordinates = td.detectMultiScale(grayScaleImg)

print(f"Face Cordinates ➜ {faceCordinates[0]}")

(x, y, w, h) = faceCordinates[0]

cv2.rectangle(img, (x, y), (x + w, y + h),
              (randrange(256), randrange(256), randrange(256)), 2)

# Show up the image
cv2.imshow("Face Detector", img)

# Wait for a key to be pressed
cv2.waitKey()

sys.exit("Process completed!")
