

from video_analyser import VideoAnalyser
from youtube import Youtube
from video_editer import VideoEditer
import re
import cv2
import numpy as np
import requests
import urllib.request
from difflib import get_close_matches


yt = Youtube()
editor =  VideoEditer()
dotaAnalyser =  VideoAnalyser()

# grab the 30 most recent games videos
teamNames, logoUrls = yt.getAllTeamNamesAndLogos()
videoIds, videoTitles = yt.get30RecentVideos('SpotnetDota2')


processedVideos = set()
for(videoId in videoIds):
    processedVideos.add(videoId)

while(true):
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
        editor.createThumbnail(team1Url, team2Url, gameNumber, 'thumbnailCreation/' + team1 + ' vs ' + team2 + ' Game ' + gameNumber + '.png') 


        # download the full video and begin video processing
        yt.downloadVideo1080p(videoUrl)

        # analyze the full video for timestamps of good clips
        clips, killsPerClip = dotaAnalyser.findClips('rawVideos/video.mp4')







def createVideo
yt.downloadVideo1080p(videoUrl)


clips, killsPerClip = dotaAnalyser.findClips('rawVideos/video.mp4')

rawClips = [[709, 734], [975, 995], [1023, 1043], [1117, 1137], [1130, 1150], [1353, 1373], [1416, 1436], [1436, 1456], [1439, 1459], [1484, 1504], [1624, 1644], [1633, 1653], [1680, 1700], [1716, 1736], [1764, 1784], [1802, 1822], [1834, 1854], [1845, 1865], [1847, 1867], [1852, 1872], [1945, 1965], [2025, 2045], [2039, 2059], [2040, 2060], [2043, 2063], [2092, 2112], [2136, 2156], [2144, 2164], [2147, 2167], [2195, 2215], [2255, 2275], [2258, 2278], [2268, 2288], [2279, 2299], [2294, 2314]]
rawKills = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
finalClips, finalKills = dotaAnalyser.fixOverlappingClips(rawClips, rawKills)

editor =  VideoEditer()

finalClips = [[709, 734], [975, 995], [1023, 1043], [1117, 1150], [1353, 1373], [1416, 1459], [1484, 1504], [1624, 1653], [1680, 1700], [1716, 1736], [1764, 1784], [1802, 1822], [1834, 1872], [1945, 1965], [2025, 2063], [2092, 2112], [2136, 2167], [2195, 2215], [2255, 2314]]

editor.combineClips()
editor.createClip(finalClips)

editor.addLogoTopRight()   
editor.combineVideoAndAudio() 
print(clips)
print(killsPerClip)


