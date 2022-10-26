
import time
import subprocess
import logging
import moviepy.editor as mp
import os
import ffmpeg
import string
from PIL import Image, ImageDraw
import urllib.request
import cv2
import numpy as np
import requests

class VideoEditer:


    def __init__(self):
        logging.basicConfig(filename='logs/editor.log', filemode='w', format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        logging.getLogger().setLevel(logging.INFO)


    def createClip(self, timeStamps):
        videoFileName = 'rawVideos/video.mp4'
        audioFileName = 'rawVideos/audio.mp4'
        fileIdentifier = ''
        for i in range(0, len(timeStamps)):
            start = time.time()
            timestamp = timeStamps[i]

            videoClipFileName = 'videoclips/videoClip{}.mp4'.format(fileIdentifier)
            audioClipFileName = 'audioclips/audioClip{}.mp4'.format(fileIdentifier)

            videoSplitCommand = ['ffmpeg', '-ss', str(timestamp[0]), '-i', videoFileName, '-t', str(int(timestamp[1]) - int(timestamp[0])), videoClipFileName]
            audioSplitCommand = ['ffmpeg', '-ss', str(timestamp[0]), '-i', audioFileName, '-t', str(int(timestamp[1]) - int(timestamp[0])), audioClipFileName]
            
            # run the ffmpeg processes
            subprocess.run(audioSplitCommand)
            subprocess.run(videoSplitCommand)

            fileIdentifier += '1'
            end = time.time()
            logging.info('Split {} in {} seconds'.format(videoClipFileName, int(end - start)))

    def addLogoTopRight(self):
        # the logo to add to the pic

        files = os.listdir('videoclips')
        outputFilenameId = ''
        for f in range(0, len(files)):
            filename = files[f]
            # load the clip and set the logo video duration to video length
            video = mp.VideoFileClip('videoclips/' + filename)

            logo = (mp.ImageClip("logos/DotaClipsLogoXWide.jpg")
                .set_duration(video.duration)
                .resize(height=65)
                .set_pos(("right","top")))

            # create the new video with logo on 
            final = mp.CompositeVideoClip([video, logo])
            outputFilename = 'logoVideoClips/logoVideo{}.mp4'.format(outputFilenameId)
            final.write_videofile(outputFilename)
            outputFilenameId += '1'

    def combineVideoAndAudio(self):
        logoFiles = os.listdir('logoVideoClips')
        outputFilenameId = '111111111111111111'

        for f in range(18, len(logoFiles)):

            videoFilename = 'logoVideoClips/' + logoFiles[f]
            fileNumber = videoFilename.replace('logoVideoClips/logoVideo', '').replace('.mp4', '')
            audioFileName = 'audioclips/audioClip{}.mp4'.format(fileNumber)
            video_stream = ffmpeg.input(videoFilename)
            audio_stream = ffmpeg.input(audioFileName)

            ffmpeg.output(audio_stream, video_stream, 'combinedClips/out{}.mp4'.format(outputFilenameId)).run()
            outputFilenameId += '1'

    def combineClips(self):
        # find *.mp4 | sed 's:\ :\\\ :g'| sed 's/^/file /' > fl.txt; sort -r fl.txt | tee sorted.txt; ffmpeg -f concat -i sorted.txt -c copy output.mp4; rm fl.txt; rm sorted.txt
        
        combineClipsCommand = ['find', 'combinedClips/*.mp4', '|', 'sed', 's:\ :\\\ :g', '|', 'sed', 's/^/file /', '>', 'combinedClips/fl.txt;', 'sort', '-r', 'combinedClips/fl.txt', '|', 'tee', 'combinedClips/sorted.txt;', 'ffmpeg', '-f', 'concat', '-i', 'combinedClips/sorted.txt', '-c', 'copy', 'combinedClips/output.mp4;', 'rm', 'combinedClips/fl.txt;', 'rm', 'combinedClips/sorted.txt']
        subprocess.run(combineClipsCommand)

        deleteClipsCommand = ['rm audioclips/*']
        subprocess.run(deleteClipsCommand)


    def createThumbnail(self, urlLeft, urlRight, gameNumber, outputFileName):

        # download the team logos
        img_data = requests.get(urlLeft).content
        with open('thumbnailCreation/leftTeam.png', 'wb') as handler:
            handler.write(img_data)
        img_data = requests.get(urlRight).content
        with open('thumbnailCreation/rightTeam.png', 'wb') as handler:
            handler.write(img_data)

        from PIL import Image
        #Read the three images
        image1 = Image.open('thumbnailCreation/leftTeam.png')
        image2 = Image.open('thumbnailCreation/rightTeam.png')
        background = Image.open('logos/background{}.PNG'.format(gameNumber))

        size1 = image1.size
        size2 = image2.size
        background_size = background.size

        # get two points from each line in the X shape to center the team logos perfectly
        topLeftPoint1 = [880, 0]
        topLeftPoint2 = [background_size[0] - 1925,background_size[1]]
        botLeftPoint1 = [880,background_size[1]]
        botLeftPoint2 = [background_size[0] - 1925,0]
        
        topRightPoint1 = [background_size[0] - 880, 0]
        topRightPoint2 = [1925, background_size[1]]
        botRightPoint1 = [background_size[0] - 880, background_size[1]]
        botRightPoint2 = [1925, 0]

        # y2-y1 / x2-x1
        topLeftm = (topLeftPoint2[1] - topLeftPoint1[1]) / (topLeftPoint2[0] - topLeftPoint1[0]) 
        botLeftm = (botLeftPoint2[1] - botLeftPoint1[1]) / (botLeftPoint2[0] - botLeftPoint1[0])
        topRightm = (topRightPoint2[1] - topRightPoint1[1]) / (topRightPoint2[0] - topRightPoint1[0]) 
        botRightm = (botRightPoint2[1] - botRightPoint1[1]) / (botRightPoint2[0] - botRightPoint1[0])

        # b = y - (m * x)
        topLeftB = topLeftPoint2[1] - (topLeftm * topLeftPoint2[0])
        botLeftB = botLeftPoint2[1] - (botLeftm * botLeftPoint2[0])
        topRightB = topRightPoint2[1] - (topRightm * topRightPoint2[0])
        botRightB = botRightPoint2[1] - (botRightm * botRightPoint2[0])

        # ratioTop height/width
        ratioLeftImage = size1[1]/size1[0]
        ratioRightImage = size2[1]/size2[0]


        leftX = (botLeftB - topLeftB) / (ratioLeftImage + topLeftm - botLeftm)
        topLeftY = (topLeftm * leftX) + topLeftB
        image1 = image1.resize((int(leftX), int(leftX * ratioLeftImage))).convert("RGBA")

        # this works because the shape is mirrored
        maxImageWidth = (botLeftB - topLeftB) / (ratioRightImage + topLeftm - botLeftm)
        rightX = background_size[0] - maxImageWidth
        topRightY = (topRightm * rightX) + topRightB
        image2 = image2.resize((int(maxImageWidth), int(maxImageWidth * ratioRightImage))).convert("RGBA")

        # create white background and past the newly sized logos. Then add background on top
        new_image = Image.new('RGB',background_size, (250,250,250))
        new_image.paste(image1,(0,int(topLeftY)), image1)
        new_image.paste(image2,(int(rightX),int(topRightY)), image2)
        new_image.paste(background, (0,0), background)
        new_image.save(outputFileName,"PNG")

