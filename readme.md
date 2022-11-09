## About
Python script to automate the creation of Dota 2 compelation videos from full games. This will first download the full video, gather the relevant timestamps of clips, split the video up at these timestamps, add the dotaClipz logo, automatically create a thumbnail, then (future update) upload the video to youtube using the Data API. 

The timestamps are gathered from the video by parsing frames of the video, using a modified binary searchish itteration for efficiency, and uses a custom CNN to read the scoreboard of each frame. When the score changes, we create a clip around it.

Using moviepy, we split that full video along the timestamps, add the dotaClips logo, and combine the clips into one full video.

The thumbnail creation is done by scraping the team logos from Liquipedia and pasting them ontop of a template created by @Allie_Leto

Eventually we will use the youtube data API to upload these compelations automatically. We can only upload 6 videos a day.
###### <sup>1</sup>: Since the projects that enable the YouTube Data API have a default quota allocation of `10,000` units per day [[2]](https://developers.google.com/youtube/v3/getting-started#calculating-quota-usage) and a video upload has a cost of approximately `1,600` units [[3]](https://developers.google.com/youtube/v3/getting-started#quota): `10,000 / 1,600 = 6.25`.


## Dependencies
The propper version of FFMPEG for ur system needs to be installed and the exe needs to either be in /bin or your path. For me this was ffmpeg-release-arm64-static.tar.xz 

Need 4 folders in base dir, "finishedVideos", "logs", "thumbnailCreation", "videoClips"

## License
[MIT](https://choosealicense.com/licenses/mit/)

