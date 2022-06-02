from typing import Optional
from joblib import load 
from fastapi import FastAPI


app = FastAPI()

vector= load('tfidf_vectors.joblib')
model = load('model.joblib')


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/prediction")
def get_prediction(review:str):
    text=[review]
    vec=vector.transform(text)
    prediction=model.predict(vec)
    prediction = int(prediction)
    if prediction>0:
        prediction ='Positive Review'
    else:
        prediction ='Negative Review'
    return{'sentence': review, 'prediction': prediction}

