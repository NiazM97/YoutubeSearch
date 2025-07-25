from fastapi import FastAPI
import polars as pl
from sentence_transformers import SentenceTransformer
from sklearn.metrics import DistanceMetric
import numpy as np
import sys
sys.path.insert(0, '/youtubesearch/app')

from functions import returnSearchResultIndexes


# define model info
model_name = 'all-MiniLM-L6-v2'
# used this model after some evaluation on the model performance
# assessed using euclidean, manhattan distances
# assessed against ranking of relevant search results against ground truth

# load model
model = SentenceTransformer(model_name)

# load video index
df = pl.scan_parquet('app/data/video-index.parquet')

# create distance metric object
dist_name = 'manhattan'
dist = DistanceMetric.get_metric(dist_name)

# create FastAPI object
app = FastAPI()

# API operations
@app.get("/")
def health_check():
    return {'health_check': 'OK'}

@app.get("/info")
def info():
    return {'name': 'yt-search', 'description': "Search API for PBS Spacetimes's YouTube videos."}

@app.get("/search")
def search(query: str):
    idx_result = returnSearchResultIndexes(query, df, model, dist)
    return df.select(['title', 'video_id']).collect()[idx_result].to_dict(as_series=False)







