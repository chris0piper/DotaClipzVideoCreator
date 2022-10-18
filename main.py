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
    


#### DOWNLOAD YOUTUBE VIDEO
# from pytube import YouTube 

# video_url="https://www.youtube.com/watch?v=yeIb87jTvIk"
# youtube = YouTube(video_url)  
# video = youtube.streams.filter(res='1080p').filter(mime_type='video/mp4').first()
# print(video.filesize)
# for stream in youtube.streams:  
#     print(stream.progressive)  
# highestQuality = youtube.streams.get_by_resolution('1080p')
# highestQuality = youtube.order_by("resolution").last()
# print(type(highestQuality))
# highestQuality.download()
# video.download()





import cv2
import pytesseract


cap = cv2.VideoCapture('EG vs BB Team Game 2  Bo2  Group Stage The International 2022 TI11  Spotnet Dota 2.mp4')
count = 65000
cap.set(cv2.CAP_PROP_POS_FRAMES, count)
ret, frame = cap.read()

while cap.isOpened():
    #   cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file      
    #   height, width = frame.shape
    shape = frame.shape
    height = shape[0]
    width = shape[1]
    # print(type(height-(height/3)))
    leftScore = frame[0:int(height/25), int(width*(4.5/10)):int(width*(4.75/10))]
    leftGray = cv2.cvtColor(leftScore, cv2.COLOR_BGR2GRAY)
    leftBw = cv2.threshold(leftGray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    # leftBw = cv2.threshold(leftGray, 220, 255, cv2.THRESH_BINARY)[1]

    rightScore = frame[0:int(height/25), int(width*(5.25/10)):int(width*(5.5/10))]
    rightGray = cv2.cvtColor(rightScore, cv2.COLOR_BGR2GRAY)
    rightBw = cv2.threshold(rightGray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    blurred = cv2.GaussianBlur(rightBw, (5, 5), 0)

    data = pytesseract.image_to_string(blurred, lang='eng', config=' - psm 11')
    print(data)
    # blurred = cv2.GaussianBlur(bw, (5, 5), 0)

    cv2.imwrite("left%d.jpg" % count, leftBw)     # save frame as JPEG file      
    cv2.imwrite("right%d.jpg" % count, rightBw)     # save frame as JPEG file      

    count += 10000
    if(count > 105100):
        break
    cap.set(cv2.CAP_PROP_POS_FRAMES, count)
    ret, frame = cap.read()

