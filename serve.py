import os
import aiohttp
import torch
import time
from PIL import Image
import argparse
from fastapi import FastAPI, HTTPException, Body

import uvicorn

from infer import Text2Image, Removebg, Image2Views, Views2Mesh, GifRenderer
from pydantic import BaseModel
import urllib
import urllib.parse

class RequestData(BaseModel):
    prompt: str
    output_dir: str

app = FastAPI()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_lite", default=False, action="store_true")
    parser.add_argument("--mv23d_cfg_path", default="./svrm/configs/svrm.yaml", type=str)
    parser.add_argument("--mv23d_ckt_path", default="weights/svrm/svrm.safetensors", type=str)
    parser.add_argument("--text2image_path", default="weights/hunyuanDiT", type=str)
    parser.add_argument("--save_folder", default="/workspace/DB", type=str)
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
rembg_model = Removebg()
# image_to_views_model = Image2Views(device=args.device, use_lite=args.use_lite)
# views_to_mesh_model = Views2Mesh(args.mv23d_cfg_path, args.mv23d_ckt_path, args.device, use_lite=args.use_lite)
text_to_image_model = Text2Image(pretrain=args.text2image_path, device=args.device, save_memory=args.save_memory)
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

async def validate(validation_url: str, timeout: int, prompt: str, DATA_DIR: str):
    async with aiohttp.ClientSession() as session:
        try:
            print(f"=================================================")
            client_timeout = aiohttp.ClientTimeout(total=float(timeout))
            
            async with session.post(validation_url, timeout=client_timeout, json={"prompt": prompt, "DATA_DIR": DATA_DIR}) as response:
                if response.status == 200:
                    result = await response.json()
                    print("Success:", result)
                else:
                    print(f"Validation failed. Please try again.: {response.status}")
                return result
        except aiohttp.ClientConnectorError:
            print(f"Failed to connect to the endpoint. Try to access again: {validation_url}.")
        except TimeoutError:
            print(f"The request to the endpoint timed out: {validation_url}")
        except aiohttp.ClientError as e:
            print(f"An unexpected client error occurred: {e} ({validation_url})")
        except Exception as e:
            print(f"An unexpected error occurred: {e} ({validation_url})")
    
    return None

def gen_3d(image, output_folder):

    res_rgba_pil = rembg_model(image)
    
    outputs = pipeline.run(
        res_rgba_pil,        
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

@app.post("/generate_from_text")
async def text_to_3d(data: RequestData):
    print(data)
    output_folder = data.output_dir
    prompt = data.prompt
    os.makedirs(output_folder, exist_ok=True)
    Extra_prompts = [
        "Angled front view, solid color background, 3d model, high quality",
        "Angled front view, solid color background, high quality, detailed textures, realistic lighting, emphasis on form and depth, suitable for 3D rendering.",
        "Angled front view, solid color background, detailed sub-components, suitable for 3D rendering, include relevant complementary objects (e.g., a stand for the clock, a decorative base for the sword) linked to the main object to create context and depth.",
    ]
    # Stage 1: Text to Image
    start = time.time()
    for extra_prompt in Extra_prompts:
        enhanced_prompt = f"{prompt}, {extra_prompt}"
        res_rgb_pil = text_to_image_model(
            enhanced_prompt,
            seed=args.t2i_seed,
            steps=args.t2i_steps
        )
        res_rgb_pil.save(os.path.join(output_folder, "img.jpg"))
        validation_url = urllib.parse.urljoin("http://127.0.0.1:8094", "/validation/")
        validation_timeout = 4
        try:
            result = await validate(validation_url=validation_url, timeout=validation_timeout, prompt=prompt, DATA_DIR=output_folder)
            # Logging
            with open(f"/workspace/fuck_prompt_history.txt", "a") as file:
                file.write(f"{prompt}-{extra_prompt}-{result['Q0']}-{result['S0']}\n")

            if result["Q0"] >= 0.4 and result["S0"] >= 0.23:
                break
            if time.time() - start > 50:
                break
        except:
            print("Failed in validation, hehehe")
    
    try:
        gen_3d(res_rgb_pil, output_folder)
        print(f"Successfully generated: {output_folder}")
        print(f"Generation time: {time.time() - start}")
        return {"success": True, "path": output_folder}
    except:
        return {"success": False, "path": output_folder}

@app.post("/generate_from_image")
async def image_to_3d(DATA_DIR: str = Body()):
    if not os.path.exists(os.path.join(DATA_DIR, "img.jpg")):
        raise HTTPException(status_code=400, detail="Image file not found")

        # Load Image
    res_rgb_pil = Image.open(os.path.join(DATA_DIR, "img.jpg"))
    gen_3d(res_rgb_pil, DATA_DIR)

    return {"success": True}


def _text_to_3d(prompt: str, output_dir: str):
    output_folder = output_dir
    os.makedirs(output_folder, exist_ok=True)

    # Stage 1: Text to Image
    start = time.time()
    res_rgb_pil = text_to_image_model(
        prompt,
        seed=args.t2i_seed,
        steps=args.t2i_steps
    )
    res_rgb_pil.save(os.path.join(output_folder, "img.jpg"))

    gen_3d(res_rgb_pil, output_folder)
    
    print(f"Successfully generated: {output_folder}")
    print(f"Generation time: {time.time() - start}")

    return {"success": True, "path": output_folder}

if __name__ == "__main__":
    # image = Image.open("bike.png")
    # gen_3d(image, "outputs")
    # prompt = "mystical sundial with constellation patterns and gem inlays"
    # extra_prompts = "Angled front view, solid color background, 3d model, high quality"
    # enhanced_prompt = f"{prompt}, {extra_prompts}"
    # _text_to_3d(enhanced_prompt, "./")
    uvicorn.run(app, host="0.0.0.0", port=args.port)




