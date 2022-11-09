## About
Python script to automate the creation of Dota 2 compelation videos from full games. This will first download the full video, gather the relevant timestamps of clips, split the video up at these timestamps, add the dotaClipz logo, automatically create a thumbnail, then (future update) upload the video to youtube using the Data API. 

The timestamps are gathered from the video by parsing frames of the video, using a modified binary searchish itteration for efficiency, and uses a custom CNN to read the scoreboard of each frame. When the score changes, we create a clip around it.

The thumbnail creation is done by scraping the team logos from Liquipedia and pasting them ontop of a template created by @Allie_Leto

## Dependencies
The propper version of FFMPEG for ur system needs to be installed and the exe needs to either be in /bin or your path. For me this was ffmpeg-release-arm64-static.tar.xz 

Need 4 folders in base dir, "finishedVideos", "logs", "thumbnailCreation", "videoClips"

## License
[MIT](https://choosealicense.com/licenses/mit/)

