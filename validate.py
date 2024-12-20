import os
import time
import numpy as np
from fastapi import HTTPException

from validation.text_clip_model import TextModel
from validation.image_clip_model import ImageModel
from validation.quality_model import QualityModel

from rendering import render, load_image

DATA_DIR = 'outputs/test'
EXTRA_PROMPT = 'anime'


text_model = TextModel()
image_model = ImageModel()
quality_model = QualityModel()


        
def init_model():
    print("loading models")
    """
    
    Loading models needed for text-to-image, image-to-image and image quality models
    After that, calculate the .glb file score
    """
    
    text_model.load_model()
    image_model.load_model()
    quality_model.load_model()

def validate(prompt: str):
    try:
        print("----------------- Validation started -----------------")
        start = time.time()
        prompt = prompt + " " + EXTRA_PROMPT
        id = 0
        
        rendered_images, before_images = render(prompt)
        
        prev_img_path = os.path.join(DATA_DIR, f"img.jpg")
        prev_img = load_image(prev_img_path)
        
        Q0 = quality_model.compute_quality(prev_img_path)
        print(f"Q0: {Q0}")
        
        S0 = text_model.compute_clip_similarity_prompt(prompt, prev_img_path) if Q0 > 0.4 else 0
        print(f"S0: {S0} - taken time: {time.time() - start}")
        if S0 < 0.23:
            return 0
            
        Ri = detect_outliers([image_model.compute_clip_similarity(prev_img, img) for img in rendered_images])
        
        Si = detect_outliers([text_model.compute_clip_similarity_prompt(prompt, before_image) for before_image in before_images])
        
        print(f"R0: taken time: {time.time() - start}")
        
        Qi = detect_outliers([quality_model.compute_quality(img) for img in before_images])
        
        S_geo = np.exp(np.log(Si).mean())
        R_geo = np.exp(np.log(Ri).mean())
        Q_geo = np.exp(np.log(Qi).mean())
        
        print("---- Rendered images similarities with preview image ---")
        print(Ri)
        print(f"R_geo: {R_geo}")
        
        print("---- Rendered images similarities with text prompt ----")
        print(Si)
        print(f"S_geo: {S_geo}")
        
        print("---- Rendered images quality ----")
        print(Qi)
        print(f"Q_geo: {Q_geo}")
        
        total_score = S0 * 0.2 + S_geo * 0.4 + R_geo * 0.3 + Q_geo * 0.1
        
        print(f"---- Total Score: {total_score} ----")
        
        if total_score < 0.35:
            return 0
        return total_score
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
    prev_img_path = os.path.join(DATA_DIR, f"img.jpg")
    
    prev_img = load_image(prev_img_path)
    print(prev_img_path)
    Q0 = quality_model.compute_quality(prev_img_path)
    print(f"Q0: {Q0}")

    # for i in range(15):
    #     prev_img_path = f"validation/output_images/image_{24*i}.png"
    #     prev_img = load_image(prev_img_path)
    #     Q0 = quality_model.compute_quality(prev_img_path)
    #     print(f"Q0: {Q0}")
    # prompt = "abandoned mining cart with rust damage and coal residue"
    prompt = "abandoned mining cart with rust damage and coal residue"
    # validate(prompt)
    # rendered_images, before_images = render(prompt)