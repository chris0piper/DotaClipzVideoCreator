

from video_analyser import VideoAnalyser
from youtube import Youtube
from video_editer import VideoEditer
import re
import cv2
import numpy as np
import requests
import urllib.request
from difflib import get_close_matches

# editor =  VideoEditer()
# editor.combineClips('cum.mp4')
# exit()
SECONDS_PER_MINUTE = 60

yt = Youtube()
editor =  VideoEditer()
dotaAnalyser =  VideoAnalyser()

# grab the 30 most recent games videos
teamNames, logoUrls = yt.getAllTeamNamesAndLogos()
videoIds, videoTitles = yt.get30RecentVideos('SpotnetDota2')


processedVideos = set()
# for(videoId in videoIds):
#     processedVideos.add(videoId)

# print(processedVideos)
while(True):
    # grab the 30 most recent games videos
    videoIds, videoTitles = yt.get30RecentVideos('SpotnetDota2')
    
    for i in range(0, len(videoTitles)):
        
        # only create compilation if its a new video
        watchId = videoIds[i]
        if(watchId in processedVideos):
            continue

        title = videoTitles[i]
        watchId = videoIds[i]
        titleSections = title.split(' | ')

        teams = titleSections[0].split(' vs ')
        team1 = re.split(' Game | game | All Games | all games | All games ', teams[0])[0]
        team2 = re.split(' Game | game ',  teams[1])[0]

        gameNumber = re.findall(r'\d+', re.split(' Game | game ', teams[1])[1])[0]
        numberOfGamesInSeries = re.findall(r'\d+', titleSections[1])[0]

        # predict which team number this title is refering to based on the titles scraped from liquipedia list
        # these are used to get the url
        predictedTeam1Name = get_close_matches(team1, teamNames, cutoff = 0.00)[0]
        predictedTeam2Name = get_close_matches(team2, teamNames, cutoff = 0.00)[0]

        team1Url = logoUrls[teamNames.index(predictedTeam1Name)]
        team2Url = logoUrls[teamNames.index(predictedTeam2Name)]

        team1Url = 'https://liquipedia.net/' + team1Url
        team2Url = 'https://liquipedia.net/' + team2Url

        thumbnailPath = 'thumbnailCreation/' + team1 + ' vs ' + team2 + ' Game ' + gameNumber + '.png'
        editor.createThumbnail(team1Url, team2Url, gameNumber, thumbnailPath) 


        # download the full video and begin video processing
        videoUrl = 'https://www.youtube.com/watch?v={}'.format(watchId)
        yt.downloadVideo1080p(videoUrl)

        # analyze the full video for timestamps of good clips + kill count
        clips, killsPerClip = dotaAnalyser.findClips('rawVideos/video.mp4')

        # split the full video into individual clips based on analysis
        editor.createClips(clips)

        # # add the dotaClipz logo to top right of video
        editor.addLogoTopRight()   

        # # combines the video clips with the audio clips
        editor.combineVideoAndAudio() 

        # combine the individual clips into one a full version
        finalVideoName = team1 + ' vs ' + team2 + ' Game ' + str(gameNumber) + ' of ' + str(numberOfGamesInSeries) + '.mp4'
        editor.combineClips(finalVideoName)



    time.sleep(10 * SECONDS_PER_MINUTE)
