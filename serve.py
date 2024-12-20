import os
import torch
import time
from PIL import Image
import argparse
from fastapi import FastAPI, HTTPException, Body
import uvicorn

from infer import Text2Image, Removebg, Image2Views, Views2Mesh, GifRenderer

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
# rembg_model = Removebg()
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

def gen_3d(image, output_folder):
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

@app.post("/generate_from_text")
async def text_to_3d(prompt: str = Body(), output_dir: str = Body()):
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




