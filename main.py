# import requests
# import re
# from bs4 import BeautifulSoup
# import pytz
# from datetime import datetime

# URL = "https://liquipedia.net/dota2/Liquipedia:Upcoming_and_ongoing_matches"
# page = requests.get(URL)

# soup = BeautifulSoup(page.content, "html.parser")

# gameList = []

# games = soup.find_all("table", class_="wikitable wikitable-striped infobox_matches_content")
# for game in games:
#     gameDict = {}
    
#     # get the playing teams
#     teams = game.find_all("span", class_="team-template-text")
#     gameDict['leftTeam'] = teams[0].text
#     gameDict['rightTeam'] = teams[1].text
#     print(gameDict['leftTeam'] + " vs. " + gameDict['rightTeam'])    

#     # get when the game starts
#     timeAndTwitch = game.find_all("span", class_="timer-object timer-object-countdown-only")
#     for element in timeAndTwitch:
#         if(element.has_attr('data-timestamp')):
#             gameDict['gametime'] = element.attrs['data-timestamp']
#             print('Series starts: ' + gameDict['gametime'])
#         if(element.has_attr('data-stream-twitch')):
#             gameDict['stream'] = element.attrs['data-stream-twitch']
#             print('On twitch channel: ' + gameDict['stream'])

#     # get number of games
#     numGames = game.find_all("abbr")
#     numGamesInts = re.findall('\d+', numGames[0].text)
#     if(len(numGamesInts) > 0):
#         gameDict['numGames'] = numGamesInts[0]
#         print('Number of games: ' + gameDict['numGames'])

#     print('\n')
#     # gameDict[]
#     # print(numGames[0].text)
    


# #### DOWNLOAD YOUTUBE VIDEO
# from pytube import YouTube 
# video_url="https://www.youtube.com/watch?v=QgHyhhHf49Q"
# youtube = YouTube(video_url)  
# video = youtube.streams.filter(res='1080p').filter(mime_type='video/mp4').first()
# print(video)
# video.download()
# exit()

# # for stream in youtube.streams:  
# #     print(stream.progressive)  
# # highestQuality = youtube.streams.get_by_resolution('1080p')
# # highestQuality = youtube.order_by("resolution").last()
# # print(type(highestQuality))
# # highestQuality.download()
# # video.download()


import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split


digits = datasets.load_digits()
_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("Training: %i" % label)

plt.savefig("plot.png")


n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

# # Create a classifier: a support vector classifier
clf = svm.SVC(gamma=0.001)

# Split data into 50% train and 50% test subsets
X_train, X_test, y_train, y_test = train_test_split(
    data, digits.target, test_size=0.5, shuffle=False
)
# print(X_train[3])
# exit()
# # Learn the digits on the train subset
clf.fit(X_train, y_train)

# # Predict the value of the digit on the test subset
# predicted = clf.predict(X_test)

# print(X_test)
# _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
# for ax, image, prediction in zip(axes, X_test, predicted):
#     ax.set_axis_off()
#     image = image.reshape(8, 8)
#     ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
#     ax.set_title(f"Prediction: {prediction}")

# plt.savefig("predict1.png")

# print(
#     f"Classification report for classifier {clf}:\n"
#     f"{metrics.classification_report(y_test, predicted)}\n"
# )

# disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
# disp.figure_.suptitle("Confusion Matrix")
# print(f"Confusion matrix:\n{disp.confusion_matrix}")

# plt.show()

# import pytesseract


import cv2
import numpy as np
import os

# images = []
# folder = 'peepeepoopoo/'
# for filename in os.listdir(folder):
#     img = cv2.imread(os.path.join(folder,filename),cv2.IMREAD_UNCHANGED)
#     # ,cv2.IMREAD_UNCHANGED
#     if img is not None:
#         images.append(img)
#     print(filename)
#     break
# print(images[0])
# exit()
cap = cv2.VideoCapture('Fnatic vs Aster Game 1  Bo2  Group Stage The International 2022 TI11  Spotnet Dota 2.mp4')
count = 65000
## end of video, loading screen
count = 166100

#start from drafting
count = 57600

cap.set(cv2.CAP_PROP_POS_FRAMES, count)
ret, frame = cap.read()

numbers = [0] * 20
heights = [0] * 20

# digitsToRead = np.zeros((64, 2000))
totalDigitCount = 0

DIGIT_WIDTH = 16

# this the end of the loading screen
# count = 16560
while cap.isOpened() and count < 270000:
    if(frame is None):
        count += 100
        cap.set(cv2.CAP_PROP_POS_FRAMES, count)
        ret, frame = cap.read()
        print("NONE FRAMExxx")
        continue
    #   cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file      
    #   height, width = frame.shape
    shape = frame.shape
    height = shape[0]
    width = shape[1]
    
    leftScore = frame[int(height/90):int(height/38), int(width*(4.5/10)):int(width*(4.75/10))]
    leftGray = cv2.cvtColor(leftScore, cv2.COLOR_BGR2GRAY)
    leftBw = cv2.threshold(leftGray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    # leftBw = cv2.threshold(leftGray, 220, 255, cv2.THRESH_BINARY)[1]

    digitShape = leftBw.shape
    digitHeight = digitShape[0]
    digitWidth = digitShape[1]

    # scan from bottom to top of column left to right, store every digit you hit
    inMiddleOfDigit = False
    xStartOfDigit = 100
    digitCount = 0
    NUMBER_OF_SLICES = 30
    for x in range(0, NUMBER_OF_SLICES):
        hitBlackMark = False
        for y in range(0, NUMBER_OF_SLICES):
            if(leftBw[int(float((digitHeight/NUMBER_OF_SLICES)) * y), int((digitWidth/NUMBER_OF_SLICES) * x)] == 0):
                hitBlackMark = True
                if(not inMiddleOfDigit):
                    xStartOfDigit = x
                break

        # this means we found a new digit. Lets save it
        if(not hitBlackMark and inMiddleOfDigit):
            currentDigit = leftBw[0:digitHeight, int((digitWidth/NUMBER_OF_SLICES)*(xStartOfDigit)):int((digitWidth/NUMBER_OF_SLICES)* (x))]

            # if the "digit" is larger than 15 pixels wide, its prob not a digit
            # we still want to save this for training data
            padded = None
            if(DIGIT_WIDTH < currentDigit.shape[1] or currentDigit.shape[1] < 4):
                xStartOfDigit = 100
                inMiddleOfDigit = False
                continue
            else:
                padded = np.pad(currentDigit, pad_width=((0, 0), (0, DIGIT_WIDTH - currentDigit.shape[1])), constant_values = 255)
            print(padded)
            print(currentDigit.shape[1])
            # break
            # # reshape from 13 by 16 to 8 by 8
            # heightScalePercent = 60 # percent of original size
            # widthScalePercent = 60 # percent of original size

            # smallHeight = int(currentDigit.shape[0] * heightScalePercent / 100)
            # smallWidth = int(currentDigit.shape[1] * widthScalePercent / 100)

            # heights[currentDigit.shape[0]] += 1
            # numbers[currentDigit.shape[1]] += 1


            # digitsToRead[totalDigitCount] = norm_image.flatten()

            # fileName = "left" + str(count) + "digit" + str(digitCount) + ".jpg"

            # split the numbers into folders based on digit width
            thisDigitsWidth = currentDigit.shape[1]
            prefix = 'unknown/'
            if(thisDigitsWidth >= 4 and thisDigitsWidth <= 7):
                prefix = '11/'
            elif(thisDigitsWidth == 8):
                prefix = '33/'
            elif(thisDigitsWidth >= 9 and thisDigitsWidth <= 11):
                prefix = '2299/'
            elif(width > 13 or width < 4):
                prefix = 'notDigit1/'

            if(thisDigitsWidth <= 4 or thisDigitsWidth >= DIGIT_WIDTH):
                xStartOfDigit = 100
                inMiddleOfDigit = False
                continue
            # prefix = 'drafting/'
            fileName = "{}FnaticVsAster{}.png".format(prefix, totalDigitCount)
            cv2.imwrite(fileName, padded)     # save frame as JPEG file 

            digitCount += 1
            xStartOfDigit = 100
            inMiddleOfDigit = False
            totalDigitCount += 1

        if(hitBlackMark):
            inMiddleOfDigit = True
        inDigitColumn = False


    ## DO THE RIGHT SIDE
    rightScore = frame[int(height/90):int(height/38), int(width*(5.25/10)):int(width*(5.5/10))]
    rightGray = cv2.cvtColor(rightScore, cv2.COLOR_BGR2GRAY)
    rightBw = cv2.threshold(rightGray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    digitShape = rightBw.shape
    digitHeight = digitShape[0]
    digitWidth = digitShape[1]

    # scan from bottom to top of column left to right, store every digit you hit
    inMiddleOfDigit = False
    xStartOfDigit = 100
    digitCount = 0
    NUMBER_OF_SLICES = 30
    for x in range(0, NUMBER_OF_SLICES):
        hitBlackMark = False
        for y in range(0, NUMBER_OF_SLICES):
            if(rightBw[int(float((digitHeight/NUMBER_OF_SLICES)) * y), int((digitWidth/NUMBER_OF_SLICES) * x)] == 0):
                hitBlackMark = True
                if(not inMiddleOfDigit):
                    xStartOfDigit = x
                break

        # this means we found a new digit. Lets save it
        if(not hitBlackMark and inMiddleOfDigit):
            currentDigit = rightBw[0:digitHeight, int((digitWidth/NUMBER_OF_SLICES)*(xStartOfDigit)):int((digitWidth/NUMBER_OF_SLICES)* (x))]

            # if the "digit" is larger than 15 pixels wide, its prob not a digit
            # we still want to save this for training data
            padded = None
            if(DIGIT_WIDTH < currentDigit.shape[1] or currentDigit.shape[1] < 4):
                xStartOfDigit = 100
                inMiddleOfDigit = False
                continue
            else:
                padded = np.pad(currentDigit, pad_width=((0, 0), (0, DIGIT_WIDTH - currentDigit.shape[1])), constant_values = 255)
            print(padded)
            print(currentDigit.shape[1])
            # break
            # # reshape from 13 by 16 to 8 by 8
            # heightScalePercent = 60 # percent of original size
            # widthScalePercent = 60 # percent of original size

            # smallHeight = int(currentDigit.shape[0] * heightScalePercent / 100)
            # smallWidth = int(currentDigit.shape[1] * widthScalePercent / 100)

            # heights[currentDigit.shape[0]] += 1
            # numbers[currentDigit.shape[1]] += 1


            # digitsToRead[totalDigitCount] = norm_image.flatten()
            totalDigitCount += 1

            # fileName = "left" + str(count) + "digit" + str(digitCount) + ".jpg"

            # split the numbers into folders based on digit width
            thisDigitsWidth = currentDigit.shape[1]
            prefix = 'unknown/'
            if(thisDigitsWidth >= 4 and thisDigitsWidth <= 7):
                prefix = '11/'
            elif(thisDigitsWidth == 8):
                prefix = '33/'
            elif(thisDigitsWidth >= 9 and thisDigitsWidth <= 11):
                prefix = '2299/'
            elif(width > 13 or width < 4):
                prefix = 'notDigit1/'

            if(thisDigitsWidth <= 4 or thisDigitsWidth >= DIGIT_WIDTH):
                break
            # prefix = 'drafting/'
            fileName = "{}FnaticVsAster{}.png".format(prefix, totalDigitCount)
            cv2.imwrite(fileName, padded)     # save frame as JPEG file 

            digitCount += 1
            xStartOfDigit = 100
            inMiddleOfDigit = False
        if(hitBlackMark):
            inMiddleOfDigit = True
        inDigitColumn = False
    # blurred = cv2.GaussianBlur(rightBw, (5, 5), 0)

    # data = pytesseract.image_to_string(blurred, lang='eng', config=' - psm 11')
    # blurred = cv2.GaussianBlur(bw, (5, 5), 0)

    # cv2.imwrite("left%d.jpg" % count, leftBw)     # save frame as JPEG file      
    # cv2.imwrite("right%d.jpg" % count, rightBw)     # save frame as JPEG file 

    count += 45

    cap.set(cv2.CAP_PROP_POS_FRAMES, count)
    ret, frame = cap.read()


norm = np.zeros((800,800))


# predicted = clf.predict(digitsToRead)
# _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))

# for ax, image, prediction in zip(axes, digitsToRead, predicted):
#     ax.set_axis_off()
#     image = image.reshape(8, 8)
#     ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
#     ax.set_title(f"Prediction: {prediction}")
# plt.savefig("predict.png")

# ninecount = 0
# previousWas1 = False
# for prediction in predicted:
#     if prediction == 9:
#         ninecount+=1
#     else:
#         ninecount = 0
#     if ninecount > 20:
#         break
