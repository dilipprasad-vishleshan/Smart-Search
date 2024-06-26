import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import csv
from io import StringIO
from pydantic import BaseModel
from typing import Union

app = FastAPI()

@app.post("/upload")
async def upload_csv(csv_file: UploadFile = File(...)):
    
    # Read the uploaded image
    contents = await csv_file.read()
    
    # Assume decoded_con/uploadtents is the string representation of CSV content
    decoded_contents = contents.decode('utf-8')

    # Use StringIO to simulate a file object 
    global csvfile
    csvfile = StringIO(decoded_contents)
    
    return "OK"


@app.put("/search/{search}")
def update_item(search : str):
    # Read CSV into a pandas DataFrame
    df = pd.read_csv(csvfile, header=0, names=['search', 'vehicle'])
    
    # Seperate it into feature and target
    X = df['search']
    y = df['vehicle']

    # Encoding the target column
    le = LabelEncoder()
    y = le.fit_transform(y)
    
    # Split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Preparing data
    tfidf = TfidfVectorizer(stop_words="english")
    X_train_tf = tfidf.fit_transform(X_train)
    #X_test_tf= tfidf.transform(X_test)
    
    # Train
    model = DecisionTreeClassifier()
    model.fit(X_train_tf, y_train)
    
    # Test
    predtext= tfidf.transform([search])
    pred = model.predict(predtext)

    # Display
    response_data = {"TATA Vehicle":list(le.inverse_transform(pred))}
    return JSONResponse(content=response_data)


