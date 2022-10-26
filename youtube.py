from pytube import YouTube 
import logging
import time
from bs4 import BeautifulSoup
import requests
import re


class Youtube:


    def __init__(self):
        logging.basicConfig(filename='logs/youtube.log', filemode='w', format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        logging.getLogger().setLevel(logging.INFO)

    def downloadVideo1080p(self, youtube_video_url):
        
        youtube = YouTube(youtube_video_url)  
        
        start = time.time()
        video = youtube.streams.filter(res='1080p').filter(mime_type='video/mp4').first()
        end = time.time()
        logging.info('Found youtube video in {}seconds'.format(int(end - start)))

        start = time.time()
        audio = youtube.streams.filter(mime_type='audio/mp4').filter(abr="128kbps").first()
        end =  time.time()
        logging.info('Found youtube audio in {}seconds'.format(int(end - start)))


        start = time.time()
        video.download(filename='rawVideos/video.mp4')
        end =  time.time()
        logging.info('Downloaded youtube video in {}seconds'.format(int(end - start)))

        start = time.time()
        audio.download(filename='rawVideos/audio.mp4')
        end =  time.time()
        logging.info('Downloaded youtube audio in {}seconds'.format(int(end - start)))


    def uploadVideo(self, filename, title, description, tags):
        print("SUDO VIDEO UPLOAD //TODO")

    def get_source(self, url):
        return BeautifulSoup(requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, verify=False).text, 'html.parser')

    # only returns non livestreams and games between two teams. pro pub games not included
    def get30RecentVideos(self, channelName):

        soup = self.get_source('https://www.youtube.com/c/SpotnetDota2/videos')
        
        watchIds = str(soup).split('"url":"/watch?v=')
        if(len(watchIds) == 0):
            return None

        # print(watchIds[1])

        finalIds = []
        videoNames = []
        # remove any streams
        for i in range(1, len(watchIds)):
            wId = watchIds[i]
            if('publishedTimeText":{"simpleText":"Streamed' in watchIds[i - 1]):
                continue
            titleString = watchIds[i - 1].split('"title":{"runs":[{"text":"')[1]
            title = titleString[:titleString.find('"')]
            if ' vs ' not in title:
                continue
            videoNames.append(title)
            finalIds.append(wId.split('"')[0])
        return finalIds, videoNames, thumbnails


    def getAllTeamNamesAndLogos(self):
        response = requests.get("https://liquipedia.net/dota2/Portal:Teams")
        soup = BeautifulSoup(response.content, 'html.parser')

        teamNames = []
        logoUrls = []

        for team in soup.find_all("span", attrs={ 'class':'team-template-lightmode'}):
            image = team.find_all('img')[0]
            teamNames.append(image.get('alt'))
            logoUrls.append(image.get('src'))
        return teamNames, logoUrls

