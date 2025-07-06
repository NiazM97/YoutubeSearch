# Youtube Search function

This project sets out to let a user semantically search phrases in a UI which will return the top 5 most relevant videos for the query, for a predefined youtube channel. 
The youtube channel is hardcoded into the app script and can't be changed by the user.
The video transcripts are accessible from an API made using FASTAPI, and the UI is made using Gradio and Huggingface.
The app uses NLP in sklearn to determine the videos to be returned by the query.
