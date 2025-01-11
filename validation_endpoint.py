import os
import time
import numpy as np
from validation.text_clip_model import TextModel
from validation.image_clip_model import ImageModel
from validation.quality_model import QualityModel
from fastapi import FastAPI, HTTPException, Body

from rendering import render, load_image
from pydantic import BaseModel
import uvicorn

EXTRA_PROMPT = 'anime'


text_model = TextModel()
image_model = ImageModel()
quality_model = QualityModel()

class RequestData(BaseModel):
    prompt: str
    DATA_DIR: str

app = FastAPI()
        
def init_model():
    print("loading models")
    """
    
    Loading models needed for text-to-image, image-to-image and image quality models
    After that, calculate the .glb file score
    """
    
    text_model.load_model()
    image_model.load_model()
    quality_model.load_model()

@app.post("/validation")
async def validate(data: RequestData):
    print(data)
    datadir = data.DATA_DIR
    prompt = data.prompt
    try:
        print("----------------- Validation started -----------------")
        start = time.time()
        prompt = prompt + " " + EXTRA_PROMPT
        id = 0
        
        prev_img_path = os.path.join(datadir, f"img.jpg")
        prev_img = load_image(prev_img_path)
        
        Q0 = quality_model.compute_quality(prev_img_path)
        print(f"Q0: {Q0}")
        
        S0 = text_model.compute_clip_similarity_prompt(prompt, prev_img_path) if Q0 > 0.4 else 0
        print(f"S0: {S0} - taken time: {time.time() - start}")
        return {"S0": S0, "Q0": Q0}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
def detect_outliers(data, threshold=1.1):
    # Calculate Q1 and Q3
    sorted_data = sorted(data)
    Q1 = np.percentile(sorted_data, 25)
    Q3 = np.percentile(sorted_data, 75)
    
    # Calculate IQR
    IQR = Q3 - Q1
    
    # Determine bounds
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    
    # Identify non-outliers
    non_outliers = [x for x in data if lower_bound <= x <= upper_bound]
    
    return non_outliers
    

if __name__ == "__main__":
    init_model()
    port = 8094
    uvicorn.run(app, host="0.0.0.0", port=port)