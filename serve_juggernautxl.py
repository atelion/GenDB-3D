from diffusers import DiffusionPipeline
import torch
from PIL import Image
import time
import os
import aiohttp
import torch
import time
from PIL import Image
import argparse
from fastapi import FastAPI, HTTPException, Body

import urllib.parse
import uvicorn

from pydantic import BaseModel

class RequestData(BaseModel):
    prompt: str
    DATA_DIR: str
    
app = FastAPI()

pipe = DiffusionPipeline.from_pretrained("RunDiffusion/Juggernaut-XL-v9", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
pipe.to("cuda")

# prompt = "space marine locker with battle damage and identification markings."
# prompt = "neon-lit arcade cabinet with retro graphics and worn control panel"
# prompt = "neon cyberpunk street sign with flickering lights and holographic display"
# extra_prompts = "Angled front view, solid color background, 3d model, detailed textures, high quality"
# enhanced_prompt = f"{prompt}, {extra_prompts}"



@app.post("/text2image")
async def image_gen(data: RequestData):
    print("====================================")
    print(data)
    output_folder = data.DATA_DIR
    prompt = data.prompt

    os.makedirs(output_folder, exist_ok=True)
    images = pipe(prompt, num_inference_steps=50).images
    images[0].save(os.path.join(output_folder, "img.jpg"))    
    return {"success": True}

def _image_gen(prompt):
    print("====================================")
    extra_prompts = "Angled front view, solid color background, 3D model, high quality"
    output_folder = "/workspace/GenDB-3D"
    images = pipe(prompt, num_inference_steps=50).images
    images[0].save(os.path.join(output_folder, "img.jpg"))    
    return {"success": True}

if __name__ == "__main__":
    port = 8095
    prompt = "crystal sword with rainbow refractions and ethereal glow"
    _image_gen(prompt)
    # uvicorn.run(app, host="0.0.0.0", port=port)
    
