# Open WebCam
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time


cap = cv2.VideoCapture(0)

# only one hand detect coz "we decide maxHands=1"
detector = HandDetector(maxHands=1)

# Define Classifier Model [Here we give the Training Model Path & labels path]
classifier = Classifier("model/keras_model.h5", "model/labels.txt")

offset = 20
imageSize = 300

# define folder-Path to save White-Image [Create folder and Save the Hand-Sign]
folder = "collectedData/A"
counter = 0

# define labels array from model [Only add those which added into keras model ]
labels = ["A", "B", "C"]

while True:
    success, img = cap.read()
    # Copy image for hiding the hand mark
    imgOutput = img.copy()

    # finding hands
    hands, img = detector.findHands(img)

    # crop image
    if hands:
        # pick only one hand so we define 0. [if we need more we simply change 1]
        hand = hands[0]

        # bounding box information
        x, y, w, h = hand['bbox']

        # take Image as squire (300 X 300) with white background
        # given value numpy unsigned-integer 8 bits [it's 0-255]
        imgWhite = np.ones((imageSize, imageSize, 3), np.uint8)*255

        # define dimension height , width for new image window
        # starting height y+ h+offset
        # starting width x+w
        imgCrop = img[y-offset: y + h+offset, x-offset: x + w+offset]

        imgCropShape = imgCrop.shape

        aspectRatio = h / w
        # condition for centering crop image inside white-window
        if aspectRatio > 1:

            # define constant k
            k = imageSize / h
            wCal = math.ceil(k*w)

            # resize image in a specific value
            imgResize = cv2.resize(imgCrop, (wCal, imageSize))

            wGap = math.ceil((imageSize - wCal)/2)
            # overlay crop image inside the white image window
            # "imgResize.shape[]" is a matrix and it has 3 values - [height , width , channel]
            imgResizeShape = imgResize.shape
            imgWhite[:, wGap:wCal + wGap] = imgResize

            # sending Classifier prediction & index value [ For predict Sign ]
            prediction, index = classifier.getPrediction(imgWhite)

            # print Prediction
            print(prediction, index)


        else:
            k = imageSize / w
            hCal = math.ceil(k * h)

            # resize image in a specific value
            imgResize = cv2.resize(imgCrop, (imageSize, hCal))
            hGap = math.ceil((imageSize - hCal) / 2)
            imgResizeShape = imgResize.shape
            imgWhite[hGap:hCal + hGap, :] = imgResize

            # sending Classifier prediction & index value [ For predict Sign ]
            prediction, index = classifier.getPrediction(imgWhite)
            # print Prediction
            print(prediction, index)

        # design bounding box
        cv2.rectangle(imgOutput, (x - offset, y - offset-50), (x - offset+100, y - offset-50+50), (255, 0, 255), cv2.FILLED)

        # prediction Sign format required [ image, labels, image size, text-font, scale, color, thickness
        cv2.putText(imgOutput, labels[index], (x, y-25), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
        # define bounding box
        cv2.rectangle(imgOutput, (x-offset, y-offset), (x + w+offset, x + h+offset), (255, 0, 255), 4)


        # # Crop image Window
        # cv2.imshow("imageCrop", imgCrop)

        # # Width background image Window
        # cv2.imshow("imgWhite", imgWhite)

    # Predict Final image window
    cv2.imshow("Image", imgOutput)
    cv2.waitKey(1)
