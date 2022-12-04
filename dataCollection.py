# Open WebCam
import  cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)

# only one hand detect coz "we decide maxHands=1"
detector = HandDetector(maxHands=1)

offset = 25
imageSize = 300

# define folder-Path to save White-Image [Create folder and Save the Hand-Sign]
folder = "collectedData/A"
counter = 0

while True:
    success, img = cap.read()

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

        aspectRatio = h/w
        # condition for centering crop image inside white-window
        if aspectRatio > 1:

            # define constant k
            k = imageSize / h
            wCal = math.ceil(k * w)

            # resize image in a specific value
            imgResize = cv2.resize(imgCrop, (wCal, imageSize))

            wGap = math.ceil((imageSize - wCal)/2)
            # overlay crop image inside the white image window
            # "imgResize.shape[]" is a matrix and it has 3 values - [height , width , channel]
            imgResizeShape = imgResize.shape
            imgWhite[:, wGap:wCal + wGap] = imgResize

        else:
            k = imageSize / w
            hCal = math.ceil(k * h)

            # resize image in a specific value
            imgResize = cv2.resize(imgCrop, (imageSize, hCal))
            hGap = math.ceil((imageSize - hCal) / 2)
            imgResizeShape = imgResize.shape
            imgWhite[hGap:hCal + hGap ,:] = imgResize



        # Crop image Window
        cv2.imshow("imageCrop", imgCrop)

        # Width background image Window
        cv2.imshow("imgWhite", imgWhite)

    # webCam image Window
    cv2.imshow("Image", img)
    key = cv2.waitKey(1)

    # define s key to save data per click
    if key == ord("s"):
        counter += 1
        # define folder & create a unique file name
        cv2.imwrite(f'{folder}/Images_{time.time()}.png', imgWhite)
        print(counter)