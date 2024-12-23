import hashlib
import os
import time
import numpy as np
from fastapi import HTTPException
import random
from validation.text_clip_model import TextModel
from validation.image_clip_model import ImageModel
from validation.quality_model import QualityModel

from rendering import render, load_image

EXTRA_PROMPT = 'anime'


text_model = TextModel()
quality_model = QualityModel()


        
def init_model():
    print("loading models")
    """
    
    Loading models needed for text-to-image, image-to-image and image quality models
    After that, calculate the .glb file score
    """
    
    text_model.load_model()
    quality_model.load_model()

def validate(prompt: str, input_folder_path: str):
    try:
        print("----------------- Validation started -----------------")
        start = time.time()
        hash_folder_name = hashlib.sha256(prompt.encode()).hexdigest()
        prompt = prompt + " " + EXTRA_PROMPT
        
        prev_img_path = os.path.join(input_folder_path, "img.jpg")
        prev_img = load_image(prev_img_path)
        
        Q0 = quality_model.compute_quality(prev_img_path)
        print(f"Q0: {Q0}")
        
        S0 = text_model.compute_clip_similarity_prompt(prompt, prev_img_path) if Q0 > 0.4 else 0
        print(f"S0: {S0} - taken time: {time.time() - start}")
        if S0 < 0.23:
            return 0
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
from infer import Text2Image
text2image_path = "weights/hunyuanDiT"
device="cuda"
save_memory = False

text_to_image_model = Text2Image(pretrain=text2image_path, device=device, save_memory=save_memory)

def text_to_image(prompt: str, t2i_seed: int, t2i_steps: int, output_folder: str):
    start = time.time()
    res_rgb_pil = text_to_image_model(
        prompt,
        seed=t2i_seed,
        steps=t2i_steps
    )
    
    res_rgb_pil.save(os.path.join(output_folder, "img.jpg"))

if __name__ == "__main__":
    output_folder = "test_dir"
    init_model()
    print("Models are initialized!!!")
    prompt = "enchanted staff with floating crystals and vine wrappings"
    start = time.time()
    
    for i in range(5):
        seed = random.randint(0, 50)
        print(seed)
        text_to_image(prompt, 31, 35, output_folder)
        score = validate(prompt, output_folder)
        print(score)
        if score != 0:
            break
                
    print(f"================================={time.time()-start}===============================================")
