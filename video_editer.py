
import time
import subprocess
import logging
import moviepy.editor as mp
from moviepy.editor import *
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


    def createClip(self, timestamp, fileId):
        
        videoFileName = 'rawVideos/video.mp4'
        audioFileName = 'rawVideos/audio.mp4'

        videoClipFileName = 'videoclips/videoClip{}.mp4'.format(fileId)
        audioClipFileName = 'audioclips/audioClip{}.wav'.format(fileId)

        start = time.time()
        
        fullVideo = VideoFileClip(videoFileName)
        fullAudio = AudioFileClip(audioFileName)

        # SPLIT THE VIDEO
        videoClip = fullVideo.subclip(timestamp[0], timestamp[1])
        audioClip = fullAudio.subclip(timestamp[0], timestamp[1])

        # ADD THE LOGO
        # load the logo at the correct length
        logo = (mp.ImageClip("logos/DotaClipsLogoXWide.jpg")
            .set_duration(videoClip.duration)
            .resize(height=65)
            .set_pos(("right","top")))
        videoClip = mp.CompositeVideoClip([videoClip, logo])

        # ADD AUDIO TO CLIP
        videoClip = videoClip.set_audio(audioClip)

        # save this subclip
        videoClip.write_videofile(videoClipFileName)

        # This tests to see if its a valid mp4 after these changes
        # error will be caught outside of this method
        vid = cv2.VideoCapture(videoClipFileName)

        end = time.time()
        logging.info('Created {} in {} seconds'.format(videoClipFileName, int(end - start)))

  
    def combineClips(self, finalFilename):
                
        # write all files in dir to .txt
        with open('videoclips/sorted.txt', 'w') as f:
            filenames = os.listdir('videoclips/') 
            for filename in filenames:
                f.write('file {}\n'.format(filename))
            f.close()
  

        # combine all the files in the dir
        os.chdir('/home/ec2-user/workspace/Twitch2Youtube/videoclips/')
        combineFiles = ['ffmpeg -f concat -i sorted.txt -c copy \"/home/ec2-user/workspace/Twitch2Youtube/finishedVideos/{}\"'.format(finalFilename)]
        subprocess.run(combineFiles, shell=True)
        os.chdir('/home/ec2-user/workspace/Twitch2Youtube/')
        
    def createThumbnail(self, urlLeft, urlRight, gameNumber, outputFileName):

        # download the team logos
        img_data = requests.get(urlLeft).content
        with open('thumbnailCreation/leftTeam.png', 'wb') as handler:
            handler.write(img_data)
        img_data = requests.get(urlRight).content
        with open('thumbnailCreation/rightTeam.png', 'wb') as handler:
            handler.write(img_data)

        #Read the three images
        image1 = Image.open('thumbnailCreation/leftTeam.png')
        image2 = Image.open('thumbnailCreation/rightTeam.png')
        background = Image.open('logos/background{}.jpeg'.format(gameNumber))

        size1 = image1.size
        size2 = image2.size
        background_size = background.size

        # get two points from each line in the X shape of the background to center/size the team logos perfectly
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
        new_image.paste(background, (0,0))
        new_image.paste(image1,(0,int(topLeftY)), image1)
        new_image.paste(image2,(int(rightX),int(topRightY)), image2)
        new_image.save(outputFileName,"jpeg")

