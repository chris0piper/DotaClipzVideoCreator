import requests
import re
from bs4 import BeautifulSoup
import pytz
from datetime import datetime

URL = "https://liquipedia.net/dota2/Liquipedia:Upcoming_and_ongoing_matches"
page = requests.get(URL)

soup = BeautifulSoup(page.content, "html.parser")

gameList = []

games = soup.find_all("table", class_="wikitable wikitable-striped infobox_matches_content")
for game in games:
    gameDict = {}
    
    # get the playing teams
    teams = game.find_all("span", class_="team-template-text")
    gameDict['leftTeam'] = teams[0].text
    gameDict['rightTeam'] = teams[1].text
    print(gameDict['leftTeam'] + " vs. " + gameDict['rightTeam'])    

    # get when the game starts
    timeAndTwitch = game.find_all("span", class_="timer-object timer-object-countdown-only")
    for element in timeAndTwitch:
        if(element.has_attr('data-timestamp')):
            gameDict['gametime'] = element.attrs['data-timestamp']
            print('Series starts: ' + gameDict['gametime'])
        if(element.has_attr('data-stream-twitch')):
            gameDict['stream'] = element.attrs['data-stream-twitch']
            print('On twitch channel: ' + gameDict['stream'])

    # get number of games
    numGames = game.find_all("abbr")
    numGamesInts = re.findall('\d+', numGames[0].text)
    if(len(numGamesInts) > 0):
        gameDict['numGames'] = numGamesInts[0]
        print('Number of games: ' + gameDict['numGames'])

    print('\n')
    # gameDict[]
    # print(numGames[0].text)
    


# print(job_elements[0])

# teams = job_elements[0].find_all("span", class_="team-template-team2-short")

# data-highlightingclass

# print(soup)