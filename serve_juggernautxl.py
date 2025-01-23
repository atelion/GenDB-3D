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
    output_dir: str
    
app = FastAPI()

pipe = DiffusionPipeline.from_pretrained("RunDiffusion/Juggernaut-XL-v9", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
pipe.to("cuda")


@app.post("/text2image")
async def image_gen(data: RequestData):
    print("=================Image Generation===================")
    print(data)
    output_folder = data.output_dir
    prompt = data.prompt

    os.makedirs(output_folder, exist_ok=True)
    images = pipe(prompt, num_inference_steps=50).images
    images[0].save(os.path.join(output_folder, "img.jpg"))    
    return {"success": True}

def _image_gen(prompt):
    print("====================================")
    # extra_prompts = "solid color background, 3D model"
    # extra_prompts = ""
    # extra_prompts = "anime"
    extra_prompts = "Angled front view, solid color background, realistic lighting, emphasis on texture and depth, suitable for 3D rendering."
    enhanced_prompt = f"{prompt}, {extra_prompts}"
    output_folder = "/workspace/GenDB-3D"
    images = pipe(enhanced_prompt, num_inference_steps=50).images
    images[0].save(os.path.join(output_folder, "img.jpg"))    
    return {"success": True}

if __name__ == "__main__":
    port = 8095
    prompt = "haunted mirror frame with tarnished silver and ghostly residue"
    _image_gen(prompt)
    # uvicorn.run(app, host="0.0.0.0", port=port)
    
