import hashlib
import os
import shutil
import time
import numpy as np
import random
from validation.text_clip_model import TextModel
from validation.image_clip_model import ImageModel
from validation.quality_model import QualityModel

from infer import Text2Image
from rendering import render, load_image


from diffusers import DiffusionPipeline
import torch
from PIL import Image
import time
import os
import torch
import time
from PIL import Image

text_model = TextModel()
quality_model = QualityModel()

EXTRA_PROMPT = 'anime'
        
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
        return S0
    except Exception as e:
        print(f"Failed in validation {e}")


def text_to_image(prompt: str, output_folder: str):
    start = time.time()
    pipe = DiffusionPipeline.from_pretrained("RunDiffusion/Juggernaut-XL-v9", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
    pipe.to("cuda")
    
    try:
        images = pipe(prompt, num_inference_steps=50).images
        images[0].save(os.path.join(output_folder, "img.jpg"))
        return True
    except Exception as e:
        print(f"Failed in generation {e}")
        return False

if __name__ == "__main__":
    bad_prompt_file = "/workspace/bad_prompts.txt"
    good_extra_file = "/workspace/good_extra_prompt.txt"
    base_folder = "/workspace/Storage"
    os.makedirs(base_folder, exist_ok=True)
    init_model()
    print("Models are initialized!!!")
    Extra_prompts = [
                "solid color background, 3D model"
                " ",
                "Angled front view, solid color background, 3d model, high quality",
                "Angled front view, solid color background, detailed sub-components, suitable for 3D rendering, include relevant complementary objects (e.g., a stand for the clock, a decorative base for the sword) linked to the main object to create context and depth.",
            ]
    start = time.time()
    
    inputfile = open("/workspace/all_prompts.txt", "r")

    lines = inputfile.readlines()
    for id, line in enumerate(lines):
        if id % 100 == 0:
            print(f"{id} th image is generated!!!!")
        print(line)
        prompt = line.strip()

        # Create a folder with the hash name
        output_folder = os.path.join(base_folder, hashlib.sha256(prompt.encode()).hexdigest())
        os.makedirs(output_folder, exist_ok=True)

        score_flag = False
        for extra_prompt in Extra_prompts:
            enhanced_prompt = f"{prompt}, {extra_prompt}"
            if text_to_image(enhanced_prompt, output_folder):
                score = validate(prompt, output_folder)
                if score == 0:                                
                    continue
                score_flag = True
                with open(good_extra_file, "a") as file:
                    file.write(f"{extra_prompt}\n")
                break
        if score_flag == False:
            with open(bad_prompt_file, "a") as file:
                file.write(f"{prompt}\n")
            
        
    print(f"================================={time.time()-start}===============================================")
