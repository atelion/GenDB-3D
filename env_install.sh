# python3.9 test success

#pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
mkdir weights
huggingface-cli download tencent/Hunyuan3D-1 --local-dir ./weights

mkdir weights/hunyuanDiT
huggingface-cli download Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled --local-dir ./weights/hunyuanDiT

pip install diffusers transformers

pip install rembg tqdm omegaconf matplotlib opencv-python imageio jaxtyping einops 

pip install SentencePiece accelerate trimesh PyMCubes xatlas libigl

pip install git+https://github.com/facebookresearch/pytorch3d

pip install git+https://github.com/NVlabs/nvdiffrast

pip install open3d

pip install fastapi

pip install uvicorn
