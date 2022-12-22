from video_analyser import VideoAnalyser
from youtube import Youtube
from video_editer import VideoEditer
from S3 import S3

import re
import cv2
import numpy as np
import requests
import urllib.request
from difflib import get_close_matches
import os
import time
import traceback

SECONDS_PER_MINUTE = 60

# when creating each clip, the videos can occational become corrupted. One solution is just
# to recreate that specific clip, however if it fails x number of times, we just exit
MAX_VIDEO_EDIT_RETRY_COUNT = 5

# location of where the raw video comes from
DOWNLOAD_LOCATION = "SpotnetDota2"

yt = Youtube()
editor =  VideoEditer()
dotaAnalyser =  VideoAnalyser()
s3 = S3()

# grab the 30 most recent games videos
teamNames, logoUrls = yt.getAllTeamNamesAndLogos()
videoIds, videoTitles = yt.get30RecentVideos(DOWNLOAD_LOCATION)


processedVideos = set()
# for videoId in videoIds:
#     processedVideos.add(videoId)

def cleanUpEditingDirectories():
    videoClipFiles = os.listdir('videoclips')
    for filename in videoClipFiles:
        os.remove('videoclips/' + filename)

    rawClipFiles = os.listdir('rawVideos')
    for filename in rawClipFiles:
        os.remove('rawVideos/' + filename)


while(True):
    # grab the 30 most recent games videos
    videoIds, videoTitles = yt.get30RecentVideos(DOWNLOAD_LOCATION)
    for i in range(len(videoTitles) - 1, 0, -1):
        
        # only create compilation if its a new video
        watchId = videoIds[i]
        if(watchId in processedVideos):
            continue
        
        title = videoTitles[i]
        print('Proccessing: ' + title)

        try:
            # only act on full length videos
            if('highlight' in title.lower()):
                processedVideos.add(watchId)
                continue
            
            # remove the spotnet logo from title
            title = re.sub(r'\|?\s?Spotnet Dota 2\s?\|?', '', title)

            # Grab the game number from the title
            matches = re.findall(r'Game \d+', title)
            gameNumber = matches[0].split()[1]
            title = re.sub(r'Game \d+', '', title)

            # Grab the number of games in series
            match = re.search(r'Bo(\d)', title)
            numberOfGamesInSeries = match.group(1)
            title = re.sub(r'Bo\d+', '', title)

            # grab the two team names
            titleSections = title.split(' | ')
            for section in titleSections:
                if bool(re.search(r'\bvs\.?\s?\b', section, re.IGNORECASE)):
                    teams = re.split(r'\bvs\.?\s?\b', section, flags=re.IGNORECASE)
                    team1 = teams[0]
                    team2 = teams[1]
        except Exception as e:
            traceback.print_exc()
            processedVideos.add(watchId)
            continue


        baseFileName = team1 + ' vs ' + team2 + ' Game ' + str(gameNumber) + ' of ' + str(numberOfGamesInSeries) 

        # predict which team name this title is refering to based on the titles scraped from liquipedia list
        # these are used to get the url
        predictedTeam1Name = get_close_matches(team1, teamNames, cutoff = 0.00)[0]
        predictedTeam2Name = get_close_matches(team2, teamNames, cutoff = 0.00)[0]

        team1Url = logoUrls[teamNames.index(predictedTeam1Name)]
        team2Url = logoUrls[teamNames.index(predictedTeam2Name)]

        team1Url = 'https://liquipedia.net/' + team1Url
        team2Url = 'https://liquipedia.net/' + team2Url

        # create the thumbnail and upload it to s3
        thumbnailPath = 'thumbnailCreation/' + baseFileName + '.jpeg'
        editor.createThumbnail(team1Url, team2Url, gameNumber, thumbnailPath) 
        s3.upload_file(thumbnailPath, 'pending-youtube-upload', 'dotaClipzThumbnails/{}.jpeg'.format(baseFileName))

        # download the full video and begin video processing
        videoUrl = 'https://www.youtube.com/watch?v={}'.format(watchId)
        yt.downloadVideo1080p(videoUrl)

        # analyze the full video for timestamps of good clips + kill count
        clips, killsPerClip = dotaAnalyser.findClips('rawVideos/video.mp4')

        if(len(clips) == 0):
            continue

        # create the video clips based on the timestamp, using a max retry num
        fileIdentifier = ''
        retryNumber = 0
        failedCreatingClips = False
        for i in range(0, len(clips)):
            # if hit max retry count, clean up all clips and go to next video
            if(retryNumber >= MAX_VIDEO_EDIT_RETRY_COUNT):
                print("\nClip {} failed to be created!\n".format(fileIdentifier))
                cleanUpEditingDirectories()
                failedCreatingClips = True
                break

            timestamp = clips[i]
            try:
                editor.createClip(timestamp, fileIdentifier)
                fileIdentifier += '1'
            except Exception as e: 
                print(e)
                retryNumber += 1
                i -= 1
            
        
        # combine all these subclips into a final video
        if(not failedCreatingClips):
            try:
                # delete file if it already exists
                finalVideoName = baseFileName + '.mp4'
                try:
                    os.remove('finishedVideos/{}'.format(finalVideoName))
                except OSError:
                    pass
                # combine all the individual clips together, then delete them
                editor.combineClips(finalVideoName)
                
                # upload this final clip to s3 to be processed by another host
                s3.upload_file('finishedVideos/{}'.format(finalVideoName), 'pending-youtube-upload', 'dotaClipz/{}'.format(finalVideoName))
                try:
                    os.remove('finishedVideos/{}'.format(finalVideoName))
                except OSError:
                    pass
                cleanUpEditingDirectories()

            except Exception as e: 
                print(e)
                cleanUpEditingDirectories()
                continue
        
        processedVideos.add(watchId)
    time.sleep(10 * SECONDS_PER_MINUTE)


