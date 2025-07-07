# Youtube Search function

This project sets out to let a user semantically search phrases in a UI which will return the top 5 most relevant videos for the query, for a predefined youtube channel (eigenchris on youtube). The youtube channel is hardcoded into the app script and can't be changed by the user. <br />
<br />
The video transcripts are accessible from an API made using FASTAPI, and the UI is made using Gradio and Huggingface.
The app uses NLP in sklearn to determine the videos to be returned by the query according to semantic distances between the query and video transcripts.


## Structure
The logic of the app is stored in the app folder. Initially I used the code in the data_pipeline folder to create the parquet files that hold the video ids, transcripts and indexes based on an API call to the Youtube Data API service. I used these files to download the video data from an arbitrary channel (eigenchris on youtube) <br />
<br />
The app.py files takes this data and then creates a FASTAPI app where a user can send a query to get the top 5 related videos to that query. Initally this FASTAPI app was stored locally, then it was deployed to a docker container where the API connection. COnnections to both of these were confimred in the jupyter notebook. <br />
<br />
Having proven the connection and that the app works, I next deployed the docker container to a GCP cloud run where I again confirmed the connection and a response to the queries. FInally, I created a UI using Gradio on Hunggingface to allow for users to interact with the search tool. The UI can be found here: https://huggingface.co/spaces/NiazM97/yt-search-project1
