import os
import torch
import time
from PIL import Image
import argparse
from fastapi import FastAPI, HTTPException, Body
import uvicorn
import hashlib
import urllib.parse
from infer import Text2Image, Removebg, Image2Views, Views2Mesh, GifRenderer
from diffusers import DiffusionPipeline
app = FastAPI()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_lite", default=False, action="store_true")
    parser.add_argument("--mv23d_cfg_path", default="./svrm/configs/svrm.yaml", type=str)
    parser.add_argument("--mv23d_ckt_path", default="weights/svrm/svrm.safetensors", type=str)
    parser.add_argument("--text2image_path", default="weights/hunyuanDiT", type=str)
    parser.add_argument("--save_folder", default="outputs/", type=str)
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--t2i_seed", default=42, type=int)
    parser.add_argument("--t2i_steps", default=25, type=int)
    parser.add_argument("--gen_seed", default=42, type=int)
    parser.add_argument("--gen_steps", default=50, type=int)
    parser.add_argument("--max_faces_num", default=90000, type=int)
    parser.add_argument("--save_memory", default=False, action="store_true")
    parser.add_argument("--do_texture_mapping", default=False, action="store_true")
    parser.add_argument("--do_render", default=False, action="store_true")
    parser.add_argument("--port", default=8093, type=int)
    return parser.parse_args()

args = get_args()

# Initialize models globally
# rembg_model = Removebg()
# image_to_views_model = Image2Views(device=args.device, use_lite=args.use_lite)
# views_to_mesh_model = Views2Mesh(args.mv23d_cfg_path, args.mv23d_ckt_path, args.device, use_lite=args.use_lite)
# text_to_image_model = Text2Image(pretrain=args.text2image_path, device=args.device, save_memory=args.save_memory)
pipe = DiffusionPipeline.from_pretrained("RunDiffusion/Juggernaut-XL-v9", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
pipe.to("cuda")
if args.do_render:
    gif_renderer = GifRenderer(device=args.device)


os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.
                                            # 'auto' is faster but will do benchmarking at the beginning.
                                            # Recommended to set to 'native' if run only once.

from TRELLIS.trellis.pipelines import TrellisImageTo3DPipeline
from TRELLIS.trellis.utils import render_utils, postprocessing_utils


pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
pipeline.cuda()

def image_to_3d(image, output_folder):
    outputs = pipeline.run(
        image,        
        seed=42,        
    )
    # GLB files can be extracted from the outputs
    glb = postprocessing_utils.to_glb(
        outputs['gaussian'][0],
        outputs['mesh'][0],
        # Optional parameters
        simplify=0.95,          # Ratio of triangles to remove in the simplification process
        texture_size=1024,      # Size of the texture used for the GLB
    )
    glb.export(os.path.join(output_folder, "mesh.glb"))

def text_to_3d(prompt: str, folder_path: str):
    
    os.makedirs(folder_path, exist_ok=True)

    # Stage 1: Text to Image
    start = time.time()
    
    images = pipe(prompt, num_inference_steps=50).images
    images[0].save(os.path.join(folder_path, "img.jpg"))    
    res_rgb_pil = images[0]
    # process_image_to_3d(res_rgb_pil, output_folder)
    image_to_3d(res_rgb_pil, folder_path)
    
    print(f"Successfully generated: {folder_path}")
    print(f"Generation time: {time.time() - start}")

    return {"success": True, "path": folder_path}


if __name__ == "__main__":
    input_file = '/workspace/update_db/warning.txt'
    db_directory = '/workspace/update_db/warndb'
    os.makedirs(db_directory, exist_ok=True)
    
    inputfile = open(input_file, "r")
    lines = inputfile.readlines()
    
    for id, line in enumerate(lines):
        if id % 10 == 0:
            print(id)
        # Remove any leading/trailing whitespace
        line = line.strip()
        if line:  # Ensure the line is not empty
            # Create a hash of the line
            line_hash = hashlib.sha256(line.encode()).hexdigest()
            # Create a folder with the hash name
            folder_path = os.path.join(db_directory, line_hash)
            os.makedirs(folder_path, exist_ok=True)
            print(f'Created folder: {folder_path}')
            # Define the path for the text file to save the line
            text_file_path = os.path.join(folder_path, 'prompt.txt')
            # Save the line in the text file
            with open(text_file_path, 'w') as text_file:
                text_file.write(line)
            
            prompt = line
            extra_prompts = "Angled front view, solid color background, 3d model, high quality"
            enhanced_prompt = f"{prompt}, {extra_prompts}"

            text_to_3d(enhanced_prompt, folder_path)
            print("-----------------------------Generated!!!------------------------------\n")
    inputfile.close()
    
    




