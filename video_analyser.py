
# Given a filename, this class will return a series of timestamps representing clips
# which it considers to be relevant. These clips would be good to be made into a compelation 

import tensorflow as tf
from keras.models import load_model
import cv2
import numpy as np
import logging
import time

# when training the model, the input data was padded to always be 16 pixels wide.
# this was larger than the largest digits and made the frame square
STANDARDIZED_DIGIT_WIDTH = 16

# the model predicts with a minimum confidence of ~.95 when given a valid digit
# anything below this threshold is random fuzz or fluff and should be ignored
THRESHOLD_DIGIT_PREDICTION_CONFIDENCE = .75

MODEL_FILE_NAME = 'model1.h5'

# when checking a scorecard for digits, we break the scorecard into 30 columns. This
# speeds up the process of scanning frames
NUMBER_OF_SLICES = 30

FRAMES_PER_SECOND = 60

# we want to parse fewer frames at the start, since we know theres no score for the first
# ~11 min of game
PRE_DRAFT_LARGE_INTERVAL = 343
PRE_DRAFT_MEDIUM_INTERVAL = 49
PRE_DRAFT_SMALL_INTERVAL = 7
PRE_DRAFT_TINY_INTERVAL = 1

# Once we get into the game, it helps having a smaller step size
POST_DRAFT_LARGE_INTERVAL = 27
POST_DRAFT_MEDIUM_INTERVAL = 9
POST_DRAFT_SMALL_INTERVAL = 3
POST_DRAFT_TINY_INTERVAL = 1

# This stores the current frame interval of the pre and post intervals
LARGEST_FRAME_INTERVAL = PRE_DRAFT_LARGE_INTERVAL * FRAMES_PER_SECOND
MIDDLE_FRAME_INTERVAL = PRE_DRAFT_MEDIUM_INTERVAL * FRAMES_PER_SECOND
SMALL_FRAME_INTERVAL = PRE_DRAFT_SMALL_INTERVAL * FRAMES_PER_SECOND
TINY_FRAME_INTERVAL = PRE_DRAFT_TINY_INTERVAL * FRAMES_PER_SECOND

LARGE_STEP = 3
MEDIUM_STEP = 2
SMALL_STEP = 1
TINY_STEP = 0

class VideoAnalyser:


    def __init__(self):
        self.digitModel = load_model(MODEL_FILE_NAME)
        logging.basicConfig(filename='logs/analysis.log', filemode='w', format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        logging.getLogger().setLevel(logging.INFO)

    def findClips(self, filename):
        
        start = time.time()
        rawClips = []
        killsPerClip = []

        # start from the videos beginning
        frameCount = 0
        lastKillCount = -1
        stepSize = LARGE_STEP
        numberOfFramesCheckedBeforeDraft = 0
        numberOfFramesCheckedAfterDraft = 0

        LARGEST_FRAME_INTERVAL = PRE_DRAFT_LARGE_INTERVAL * FRAMES_PER_SECOND
        MIDDLE_FRAME_INTERVAL = PRE_DRAFT_MEDIUM_INTERVAL  * FRAMES_PER_SECOND
        SMALL_FRAME_INTERVAL = PRE_DRAFT_SMALL_INTERVAL  * FRAMES_PER_SECOND
        TINY_FRAME_INTERVAL = PRE_DRAFT_TINY_INTERVAL  * FRAMES_PER_SECOND

        drafting = True

        # itterate through the video first at the largest interval,
        # if a kill occured, backtrack and continue again at the medium interval
        # if a kill occured, backtrack and continue again at the samalled interval until the kill is found
        cap = cv2.VideoCapture(filename)
        while cap.isOpened():

            if(drafting):
                numberOfFramesCheckedBeforeDraft += 1
            else:
                numberOfFramesCheckedAfterDraft += 1


            cap.set(cv2.CAP_PROP_POS_FRAMES, frameCount)
            ret, frame = cap.read()

            scores = self.getScoreFromFrame(frame)

            currentScore = lastKillCount
            
            if scores != None:
                currentScore = scores[0] + scores[1]

            # no score and not drafting means we hit end of game
            elif (not drafting):
                # go backwards till we hit end of game
                if(stepSize is LARGE_STEP):
                    frameCount -= LARGEST_FRAME_INTERVAL
                    stepSize = MEDIUM_STEP
                elif(stepSize is MEDIUM_STEP):
                    frameCount -= MIDDLE_FRAME_INTERVAL
                    stepSize = SMALL_STEP
                elif(stepSize is SMALL_STEP):
                    frameCount -= SMALL_FRAME_INTERVAL
                    stepSize = TINY_STEP
                # means we just hit the end of the game. we can break loop
                else:
                    currentTime = int(frameCount / FRAMES_PER_SECOND)
                    rawClips.append([currentTime - 20, currentTime])
                    killsPerClip.append(currentScore - lastKillCount)
                    print('GAME JUST ENDED: Time - ' + str(int(frameCount / 60 / 60)) + ':' + str(int((frameCount/60)%60)))

                    break

            # if no score was found or it remained the same, keep looking at the current step sie
            if(currentScore <= lastKillCount):
                if(stepSize is LARGE_STEP):
                    frameCount += LARGEST_FRAME_INTERVAL
                elif(stepSize is MEDIUM_STEP):
                    frameCount += MIDDLE_FRAME_INTERVAL
                elif(stepSize is SMALL_STEP):
                    frameCount += SMALL_FRAME_INTERVAL
                else:
                    frameCount += TINY_FRAME_INTERVAL
                continue
            
            # if a score was found, back up and downgrade step sizes
            else:
                if(stepSize is LARGE_STEP):
                    frameCount -= LARGEST_FRAME_INTERVAL
                    stepSize = MEDIUM_STEP
                elif(stepSize is MEDIUM_STEP):
                    frameCount -= MIDDLE_FRAME_INTERVAL
                    stepSize = SMALL_STEP
                elif(stepSize is SMALL_STEP):
                    frameCount -= SMALL_FRAME_INTERVAL
                    stepSize = TINY_STEP
                # means a kill happened at this time
                else:
                    # The exact second of the start of the game
                    currentTime = int(frameCount / FRAMES_PER_SECOND)
                    if(drafting): 
                        rawClips.append([max(0, currentTime - 25), currentTime])
                        killsPerClip.append(0)

                        # once the game starts, we want to take smaller steps through video
                        LARGEST_FRAME_INTERVAL = POST_DRAFT_LARGE_INTERVAL * FRAMES_PER_SECOND
                        MIDDLE_FRAME_INTERVAL = POST_DRAFT_MEDIUM_INTERVAL  * FRAMES_PER_SECOND
                        SMALL_FRAME_INTERVAL = POST_DRAFT_SMALL_INTERVAL  * FRAMES_PER_SECOND
                        TINY_FRAME_INTERVAL = POST_DRAFT_TINY_INTERVAL  * FRAMES_PER_SECOND
                        drafting = False
                        print('DRAFT JUST ENDED: Time - ' + str(int(frameCount / 60 / 60)) + ':' + str(int((frameCount/60)%60)) + " Score - " + str(scores[0]) + " to " + str(scores[1]))

                    else:
                        rawClips.append([currentTime - 10, currentTime + 10])
                        killsPerClip.append(currentScore - lastKillCount)
                        print('Time - ' + str(int(frameCount / 60 / 60)) + ':' + str(int((frameCount/60)%60)) + " Score - " + str(scores[0]) + " to " + str(scores[1]))

                    lastKillCount = currentScore
                    stepSize = LARGE_STEP
                        
        end = time.time()
        logging.info('Video analyzed for optimal timestamps in {} seconds. Number of frames BEFORE DRAFT: {} Number of frames AFTER DRAFT: {}  TOTAL: {}'.format(int(end - start), numberOfFramesCheckedBeforeDraft, numberOfFramesCheckedAfterDraft, numberOfFramesCheckedAfterDraft + numberOfFramesCheckedBeforeDraft))

        return self.fixOverlappingClips(rawClips, killsPerClip)

    def getScoreFromFrame(self, frame):

        # videos might return frames as none when we get to the end
        if frame is None:
            return None

        shape = frame.shape
        height = shape[0]
        width = shape[1]

        # crop the full screen frame into being the two score cards
        # then convert them to black and white
        leftScore = frame[int(height/90):int(height/38), int(width*(4.5/10)):int(width*(4.75/10))]
        leftGray = cv2.cvtColor(leftScore, cv2.COLOR_BGR2GRAY)
        leftBw = cv2.threshold(leftGray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        rightScore = frame[int(height/90):int(height/38), int(width*(5.25/10)):int(width*(5.5/10))]
        rightGray = cv2.cvtColor(rightScore, cv2.COLOR_BGR2GRAY)
        rightBw = cv2.threshold(rightGray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        leftScore = self.extractDigitFromScorecard(leftBw)
        if( leftScore is None):
            return None
        rightScore = self.extractDigitFromScorecard(rightBw)
        if( rightScore is None):
            return None

        return [leftScore, rightScore]


    def extractDigitFromScorecard(self, scorecard):
        shape = scorecard.shape
        height = shape[0]
        width = shape[1]

        toPredict = []

        # moving from left to right, we check each column every NUMBER_OF_SLICES 
        # to see if it contains any colored dots, indicating a digit.
        # we then split the larger scorecard into each individual digit
        inMiddleOfDigit = False
        xStartOfDigit = 100
        for x in range(0, NUMBER_OF_SLICES):

            hitBlackMark = False
            
            # check if a black mark exists in this column
            for y in range(0, NUMBER_OF_SLICES):
                if(scorecard[int(float((height/NUMBER_OF_SLICES)) * y), int((width/NUMBER_OF_SLICES) * x)] == 0):
                    hitBlackMark = True
                    if(not inMiddleOfDigit):
                        xStartOfDigit = x
                    break

            # this means we found a new digit. Lets analyse it
            if(not hitBlackMark and inMiddleOfDigit):
                
                currentDigit = scorecard[0:height, int((width/NUMBER_OF_SLICES)*(xStartOfDigit)):int((width/NUMBER_OF_SLICES)* (x))]
                
                # if the "digit" is larger than 15 pixels wide, or narrower than 4, 
                # its prob fuzz and not a digit
                padded = None
                currentDigitsWidth = currentDigit.shape[1]
                if(currentDigitsWidth <= 4 or currentDigitsWidth >= STANDARDIZED_DIGIT_WIDTH):
                    xStartOfDigit = 100
                    inMiddleOfDigit = False
                    continue
                else:
                    padded = np.pad(currentDigit, pad_width=((0, 0), (0, STANDARDIZED_DIGIT_WIDTH - currentDigit.shape[1])), constant_values = 255)

                toPredict.append(padded)

                xStartOfDigit = 100
                inMiddleOfDigit = False

            if(hitBlackMark):
                inMiddleOfDigit = True
            inDigitColumn = False

        if(len(toPredict) == 0):
            return None
        # shape the values into what the model was trained on
        toPredict = np.array(toPredict)
        toPredict = toPredict.reshape((toPredict.shape[0], 16, 16, 1))
        toPredict = toPredict.astype('float32')
        toPredict = toPredict / 255.0

        # predict each digit
        prediction = self.digitModel.predict(toPredict, verbose=0)

        # concat these ints together
        finalPrediction = ""
        for prediction in prediction:
            predictedDigit = np.argmax(prediction)

            # durring drafting/when games are done, scropped area might have random fuzz
            # the model is normally predicting with ~95% confidence at a minimum, so
            # anything that it is less than ~.75% confidence with can be assumed to be
            # garbage fuzz
            if(prediction[predictedDigit] < THRESHOLD_DIGIT_PREDICTION_CONFIDENCE):
                return None
            
            finalPrediction += str(np.argmax(prediction))
        return int(finalPrediction)




    def fixOverlappingClips(self, rawClips, rawKills):

        finalClips = []
        finalKills = []

        lastClip = [-100, -100]
        lastKill = 0

        largerClip = []
        largerClipKillCount = 0

        inBigClip = False

        for i in range(0, len(rawClips)):
            timestamp = rawClips[i]
            killcount = rawKills[i]
            # if clips are close to eachother or overlapping, combine
            if(timestamp[0] - 10 < lastClip[1]):
                if inBigClip:
                    largerClip[1] = timestamp[1]
                    largerClipKillCount += killcount
                else:
                    finalClips.pop()
                    finalKills.pop()
                    largerClip = [lastClip[0], timestamp[1]]
                    largerClipKillCount = lastKill + killcount
                    inBigClip = True
            #
            else:
                if inBigClip:
                    finalClips.append(largerClip)
                    finalKills.append(largerClipKillCount)
                    largerClip = []
                    largerClipKillCount = 0
                    inBigClip = False
                finalClips.append(timestamp)
                finalKills.append(killcount)
            lastClip = timestamp
            lastKill = killcount
        if inBigClip:
            finalClips.append(largerClip)
            largerClip = []

        return finalClips, finalKills
