git clone --recurse-submodules https://github.com/microsoft/TRELLIS.git
# Read Arguments
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121

# Get system information
WORKDIR=$(pwd)
PYTORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda)")
CUDA_MAJOR_VERSION=$(echo $CUDA_VERSION | cut -d'.' -f1)
CUDA_MINOR_VERSION=$(echo $CUDA_VERSION | cut -d'.' -f2)
echo "[SYSTEM] PyTorch Version: $PYTORCH_VERSION, CUDA Version: $CUDA_VERSION"
pip install pillow imageio imageio-ffmpeg tqdm easydict opencv-python-headless scipy ninja rembg onnxruntime trimesh xatlas pyvista pymeshfix igraph transformers
pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8

# Xformer
echo "------------------xformer------------------"
pip install xformers==0.0.24 --index-url https://download.pytorch.org/whl/cu121

# FLASHATTN
echo "------------------FLASHATTN------------------"
pip install flash-attn===1.0.4

# KAOLIN
echo "------------------KAOLIN------------------"
pip install kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.2.0_cu118.html

# NVDIFFRAST
echo "------------------NVDIFFRAST------------------"
mkdir -p /tmp/extensions
git clone https://github.com/NVlabs/nvdiffrast.git /tmp/extensions/nvdiffrast
pip install /tmp/extensions/nvdiffrast

# DIFFOCTREERAST
echo "--------------------DIFFOCTREERAST------------------"
mkdir -p /tmp/extensions
git clone --recurse-submodules https://github.com/JeffreyXiang/diffoctreerast.git /tmp/extensions/diffoctreerast
pip install /tmp/extensions/diffoctreerast

# MIPGAUSSIAN
echo "------------------MIPGAUSSIAN------------------"
mkdir -p /tmp/extensions
git clone https://github.com/autonomousvision/mip-splatting.git /tmp/extensions/mip-splatting
pip install /tmp/extensions/mip-splatting/submodules/diff-gaussian-rasterization/

# VOX2SEQ
echo "------------------VOX2SEQ------------------"
mkdir -p /tmp/extensions
cp -r extensions/vox2seq /tmp/extensions/vox2seq
pip install /tmp/extensions/vox2seq

# SPCONV
echo "------------------SPCONV------------------"
pip install spconv-cu120

pip install gradio==4.44.1 gradio_litmodel3d==0.0.1

