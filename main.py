# import requests
# import re
# from bs4 import BeautifulSoup
# import pytz
# from datetime import datetime

# 128 kbps

# from pytube import YouTube 
# video_url="https://www.youtube.com/watch?v=4ypo9cx63VM"
# youtube = YouTube(video_url)  
# video = youtube.streams.filter(res='1080p').filter(mime_type='video/mp4').first()
# audio = youtube.streams.filter(mime_type='audio/mp4').filter(abr="128kbps").first()
# print(video)
# print(audio)
# video.download(filename='video.mp4')
# audio.download(filename='audio.mp4')
# exit()

# import datetime
# from google import Create_Service
# from googleapiclient.http import MediaFileUpload

# CLIENT_SECRET_FILE = 'youtubeSecret.json'
# API_NAME = 'youtube'
# API_VERSION = 'v3'
# SCOPES = ['https://www.googleapis.com/auth/youtube.upload']

# service = Create_Service(CLIENT_SECRET_FILE, API_NAME, API_VERSION, SCOPES)

# exit()
# import ffmpeg

# video_stream = ffmpeg.input('videoclips/output.mp4')
# audio_stream = ffmpeg.input('audioclips/output.mp4')
# ffmpeg.output(audio_stream, video_stream, 'out.mp4').run()
# exit()


# from simple_youtube_api.Channel import Channel
# from simple_youtube_api.LocalVideo import LocalVideo

# # loggin into the channel
# channel = Channel()
# channel.login("youtubeSecret.json", "credentials.storage")

# video = LocalVideo(file_path="watermarked.mp4")

# video.set_title("My Title")
# video.set_description("This is a description")
# video.set_tags(["this", "tag"])
# video.set_category("gaming")
# video.set_default_language("en-US")

# video.set_embeddable(True)
# video.set_license("creativeCommon")
# video.set_privacy_status("public")
# video.set_public_stats_viewable(True)

# video.set_thumbnail_path('DotaClipzLogoHorizontal.jpg')

# video = channel.upload_video(video)
# print(video.id)
# print(video)
# video.like()
# exit()


# from youtube_upload.client import YoutubeUploader

# uploader = YoutubeUploader('296551375650-helnsv7ltoeo0o9rvstm64blffspos7i.apps.googleusercontent.com','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')

# uploader.authenticate()

# options = {
#     "title" : "Hello World!", # The video title
#     "description" : "hoping this works", # The video description
#     "tags" : ["tag1", "tag2", "tag3"],
#     "categoryId" : "20",
#     "privacyStatus" : "public", # Video privacy. Can either be "public", "private", or "unlisted"
#     "kids" : False, # Specifies if the Video if for kids or not. Defaults to False.
#     # "thumbnailLink" : "https://cdn.havecamerawilltravel.com/photographer/files/2020/01/youtube-logo-new-1068x510.jpg" # Optional. Specifies video thumbnail.
# }

# # upload video
# uploader.upload(file_path, options) 
# exit()


import moviepy.editor as mp

# my_clip = VideoFileClip("videoclips/videoClip1111111111.mp4", audio=False)  #  The video file with audio enabled

video = mp.VideoFileClip("out.mp4")

logo = (mp.ImageClip("DotaClipsLogoXWide.jpg")
          .set_duration(video.duration)
          .resize(height=65) # if you need to resize...
        #   .margin(right=8, top=8, opacity=0) # (optional) logo-border padding
          .set_pos(("right","top")))

final = mp.CompositeVideoClip([video, logo])
final.write_videofile("logoOnCompelition.mp4")
exit()


import time
import subprocess

with open('readme.txt', 'w') as f:

    startTime = time.time()
    lastSplit = startTime
    f.write(str(int(startTime)) + '\n')

    
    finalTimeStamps = [[801, 821], [1059, 1079], [1161, 1184], [1200, 1220], [1239, 1259], [1305, 1361], [1419, 1451], [1491, 1511], [1536, 1586], [1683, 1703], [1725, 1745], [1770, 1799], [1812, 1832], [1923, 1946], [1989, 2033], [2073, 2108], [2163, 2192], [2289, 2309], [2334, 2354], [2451, 2489]]
    # 00:00:00 -t 00:50:00
    # finalTimeStamps = [['00:00:00', '00:00:05']]
    videoFileName = 'video.mp4'
    audioFileName = 'audio.mp4'
    fileIdentifier = '11111111'
    for i in range(8, len(finalTimeStamps)):
        timestamp = finalTimeStamps[i]
        videoClipFileName = 'videoclips/videoClip{}.mp4'.format(fileIdentifier)
        audioClipFileName = 'audioclips/audioClip{}.mp4'.format(fileIdentifier)
        # command = ['/bin/ffmpeg/ffmpeg', '-i', videoFileName, '-ss', str(timestamp[0]), '-t', str(timestamp[1]), clipFileName]
        #working command
        # /bin/ffmpeg/ffmpeg -i 'EG vs Thunder Awaken Game 2  Bo3  Upper Bracket The International 2022 TI11  Spotnet Dota 2.mp4' -ss 1 -t 13 clips/clip1.mp4
        videoSplitCommand = ['ffmpeg', '-ss', str(timestamp[0]), '-i', videoFileName, '-t', str(int(timestamp[1]) - int(timestamp[0])), videoClipFileName]
        audioSplitCommand = ['ffmpeg', '-ss', str(timestamp[0]), '-i', audioFileName, '-t', str(int(timestamp[1]) - int(timestamp[0])), audioClipFileName]
        
        # run the ffmpeg processes
        subprocess.run(audioSplitCommand)
        subprocess.run(videoSplitCommand)

        fileIdentifier += '1'
        # log time
        splitTime = time.time()
        f.write(str(int(splitTime-lastSplit)) + '\n')
        lastSplit = splitTime
    endTime = time.time()
    f.write(str(int(endTime-lastSplit)) + '\n')
exit()


# clips = []
# for frame in highlight_frames:
# clip_name = video.subclip(round(frame/fps) - 4, round(frame/fps)+ 2)
# clips.append(clip_name)
# final_clip = concatenate_videoclips(clips)
# final_clip.write_videofile(output_path,
#                            codec='libx264',
#                            audio_codec='aac',
#                            fps=fps)



# find *.mp4 | sed 's:\ :\\\ :g'| sed 's/^/file /' > fl.txt; /bin/ffmpeg/ffmpeg -f concat -i fl.txt -c copy output.mp4; rm fl.txt



import os
import cv2
import numpy as np
# folder = '2299/'
# files = os.listdir(folder)
# draftX = []
# for f in range(0, len(files)):
#     filename = files[f]
#     img = cv2.imread(os.path.join(folder,filename),cv2.IMREAD_UNCHANGED)
#     if img is not None:
#             draftX.append(img)
#     f += 100

# draftX = np.array(draftX)
# draftX = draftX.reshape((draftX.shape[0], 16, 16, 1))
# draftX = draftX.astype('float32')
# draftX = draftX / 255.0

import tensorflow as tf
from keras.models import load_model
model = load_model('model1.h5')
# print(model)
# predict_value = model.predict(draftX)
# print(np.argmax(model.predict(draftX)))
# draftX = np.array(draftX)
# draftX = draftX.reshape((draftX.shape[0], 16, 16, 1))
# draftX = draftX.astype('float32')
# draftX = draftX / 255.0
# print(draftX.shape)
# for prediction in model.predict(draftX):
#     print(np.argmax(prediction))
# print(model)



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
    


# # #### DOWNLOAD YOUTUBE VIDEO
# from pytube import YouTube 
video_url="https://www.youtube.com/watch?v=4ypo9cx63VM"
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


# digits = datasets.load_digits()
# _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
# for ax, image, label in zip(axes, digits.images, digits.target):
#     ax.set_axis_off()
#     ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
# #     ax.set_title("Training: %i" % label)

# # plt.savefig("plot.png")


# n_samples = len(digits.images)
# data = digits.images.reshape((n_samples, -1))

# # # Create a classifier: a support vector classifier
# clf = svm.SVC(gamma=0.001)

# # Split data into 50% train and 50% test subsets
# X_train, X_test, y_train, y_test = train_test_split(
#     data, digits.target, test_size=0.5, shuffle=False
# )
# # print(X_train[3])
# # exit()
# # # Learn the digits on the train subset
# clf.fit(X_train, y_train)

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

# os.environ["IMAGEIO_FFMPEG_EXE"] = "lib/python3.7/site-packages/ffmpeg"



# clip =   VideoFileClip("EG vs Thunder Awaken Game 2  Bo3  Upper Bracket The International 2022 TI11  Spotnet Dota 2.mp4")
cap = cv2.VideoCapture('EG vs Thunder Awaken Game 2  Bo3  Upper Bracket The International 2022 TI11  Spotnet Dota 2.mp4')


numbers = [0] * 20
heights = [0] * 20

# digitsToRead = np.zeros((64, 2000))
totalDigitCount = 0

DIGIT_WIDTH = 16

DRAFT_START = 48060

# count = 39600
count = 47400
cap.set(cv2.CAP_PROP_POS_FRAMES, count)
ret, frame = cap.read()
lastKillCount = 0
FRAME_INTERVAL = 180
clipTimestamps = []
# 151200
while cap.isOpened() and count < 151200:
    print("Time- " + str(int(count / (60 * 60))) + ':' + str(int((count / 60) % 60)))

    # print(count)
    toPredictLeft = []
    toPredictRight = []
    if(frame is None):
        count += FRAME_INTERVAL
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
            # print(padded)
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
            # if(count < DRAFT_START):
            #     prefix = 'drafting/'
            # elif(thisDigitsWidth >= 4 and thisDigitsWidth <= 7):
            #     prefix = '11/'
            # elif(thisDigitsWidth == 8):
            #     prefix = '33/'
            # elif(thisDigitsWidth >= 9 and thisDigitsWidth <= 11):
            #     prefix = '2299/'
            # elif(width > 13 or width < 4):
            #     prefix = 'notDigit1/'

            if(thisDigitsWidth <= 4 or thisDigitsWidth >= DIGIT_WIDTH):
                xStartOfDigit = 100
                inMiddleOfDigit = False
                continue
            # prefix = 'drafting/'
            # print(padded)
            toPredictLeft.append(padded)
            # fileName = "{}SecretVsPSG{}.png".format(prefix, totalDigitCount)
            # cv2.imwrite(fileName, padded)     # save frame as JPEG file 

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
            # print(padded)
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
            # prefix = 'unknown/'
            # if(count < DRAFT_START):
            #     prefix = 'drafting/'
            # elif(thisDigitsWidth >= 4 and thisDigitsWidth <= 7):
            #     prefix = '11/'
            # elif(thisDigitsWidth == 8):
            #     prefix = '33/'
            # elif(thisDigitsWidth >= 9 and thisDigitsWidth <= 11):
            #     prefix = '2299/'
            # elif(width > 13 or width < 4):
            #     prefix = 'notDigit1/'

            if(thisDigitsWidth <= 4 or thisDigitsWidth >= DIGIT_WIDTH):
                break
            # prefix = 'drafting/'
            toPredictRight.append(padded)
            # fileName = "{}SecretVsPSG{}.png".format(prefix, totalDigitCount)
            # cv2.imwrite(fileName, padded)     # save frame as JPEG file 

            digitCount += 1
            xStartOfDigit = 100
            inMiddleOfDigit = False
            totalDigitCount += 1
        if(hitBlackMark):
            inMiddleOfDigit = True
        inDigitColumn = False
    # make predictions
    if(len(toPredictLeft) == 0 or len(toPredictRight) == 0):
        count += FRAME_INTERVAL
        continue
    toPredictLeft = np.array(toPredictLeft)
    toPredictLeft = toPredictLeft.reshape((toPredictLeft.shape[0], 16, 16, 1))
    toPredictLeft = toPredictLeft.astype('float32')
    toPredictLeft = toPredictLeft / 255.0

    toPredictRight = np.array(toPredictRight)
    toPredictRight = toPredictRight.reshape((toPredictRight.shape[0], 16, 16, 1))
    toPredictRight = toPredictRight.astype('float32')
    toPredictRight = toPredictRight / 255.0

    # print(toPredictLeft.shape)
    leftPredict = model.predict(toPredictLeft, verbose=0)
    rightPredict = model.predict(toPredictRight, verbose=0)
    
    endPredictLeft = ""
    for prediction in leftPredict:
        endPredictLeft += str(np.argmax(prediction))

    endPredictRight = ""
    for prediction in rightPredict:
        endPredictRight += str(np.argmax(prediction))

    currentKillCount = int(endPredictLeft) + int(endPredictRight)
    if(lastKillCount < currentKillCount):
        print(endPredictLeft + " to " + endPredictRight)
        print('-------------------------------------------------------------')
        lastKillCount = currentKillCount
        timestamps = [int(count/60) - 10, int(count/60) + 10]
        clipTimestamps.append(timestamps)
    # data = pytesseract.image_to_string(blurred, lang='eng', config=' - psm 11')
    # blurred = cv2.GaussianBlur(bw, (5, 5), 0)

    # cv2.imwrite("left%d.jpg" % count, leftBw)     # save frame as JPEG file      
    # cv2.imwrite("right%d.jpg" % count, rightBw)     # save frame as JPEG file 

    count += FRAME_INTERVAL

    cap.set(cv2.CAP_PROP_POS_FRAMES, count)
    ret, frame = cap.read()


finalTimeStamps = []
lastClip = [-100, -100]
largerClip = []
inBigClip = False

print(clipTimestamps)
for timestamp in clipTimestamps:
    # this will combine clips that are close to eachother
    if(timestamp[0] - 10 < lastClip[1]):
        if inBigClip:
            largerClip[1] = timestamp[1]
        else:
            finalTimeStamps.pop()
            largerClip = [lastClip[0], timestamp[1]]
            inBigClip = True
    else:
        if inBigClip:
            finalTimeStamps.append(largerClip)
            largerClip = []
            inBigClip = False
        finalTimeStamps.append(timestamp)
    lastClip = timestamp
if inBigClip:
    finalTimeStamps.append(largerClip)
    largerClip = []
print(finalTimeStamps)

clipTimestamps = [[801, 821], [1059, 1079], [1161, 1181], [1164, 1184], [1200, 1220], [1239, 1259], [1305, 1325], [1320, 1340], [1341, 1361], [1419, 1439], [1431, 1451], [1491, 1511], [1536, 1556], [1539, 1559], [1542, 1562], [1551, 1571], [1566, 1586], [1683, 1703], [1725, 1745], [1770, 1790], [1776, 1796], [1779, 1799], [1812, 1832], [1923, 1943], [1926, 1946], [1989, 2009], [2010, 2030], [2013, 2033], [2073, 2093], [2079, 2099], [2082, 2102], [2085, 2105], [2088, 2108], [2163, 2183], [2169, 2189], [2172, 2192], [2289, 2309], [2334, 2354], [2451, 2471], [2457, 2477], [2463, 2483], [2466, 2486], [2469, 2489]]
finalTimeStamps = [[801, 821], [1059, 1079], [1161, 1184], [1200, 1220], [1239, 1259], [1305, 1361], [1419, 1451], [1491, 1511], [1536, 1586], [1683, 1703], [1725, 1745], [1770, 1799], [1812, 1832], [1923, 1946], [1989, 2033], [2073, 2108], [2163, 2192], [2289, 2309], [2334, 2354], [2451, 2489]]


# import subprocess
# videoFileName = 'EG vs Thunder Awaken Game 2  Bo3  Upper Bracket The International 2022 TI11  Spotnet Dota 2.mp4'
# for i in range(0, len(finalTimeStamps)):
#     timestamp = finalTimeStamps[i]
#     clipFileName = 'clips/clip{}.mp4'.format(i)
#     cmp = ['ffmpeg', '-i', videoFileName, '-ss', timestamp[0], '-t' timestamp[1], clipFileName]
#     subprocess.run(cmp)
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
