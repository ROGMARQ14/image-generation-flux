# FLUX AI Image Generator (Streamlit Edition)

This app provides an interactive web interface for generating images using the FLUX model via ComfyUI, powered by Streamlit.

## Features

- **Prompt field** for text-to-image generation
- **Aspect ratio** selection with auto-calculated height
- **Custom width** input
- **Seed** control for reproducibility
- **Steps** setting for image quality
- **Live preview** of generated images
- **Edit field** to refine images with new prompts (using the same seed)
- **Download** button for saving generated images

## Setup Instructions

### 1. Clone the repository

git clone -b totoro3 https://github.com/camenduru/ComfyUI TotoroUI
cd TotoroUI

### 2. Download Model Files

mkdir -p models/unet models/vae models/clip
wget https://huggingface.co/camenduru/FLUX.1-dev/resolve/main/flux1-dev-fp8.safetensors -O models/unet/flux1-dev-fp8.safetensors
wget https://huggingface.co/camenduru/FLUX.1-dev/resolve/main/ae.sft -O models/vae/ae.sft
wget https://huggingface.co/camenduru/FLUX.1-dev/resolve/main/clip_l.safetensors -O models/clip/clip_l.safetensors
wget https://huggingface.co/camenduru/FLUX.1-dev/resolve/main/t5xxl_fp8_e4m3fn.safetensors -O models/clip/t5xxl_fp8_e4m3fn.safetensors
wget https://huggingface.co/camenduru/FLUX.1-dev/resolve/main/test.png -O test.png

### 3. Install Python dependencies

pip install -r requirements.txt

text

### 4. Run the Streamlit app

streamlit run app.py

## Notes

- Make sure you have a GPU available for best performance.
- Place `app.py` in the root of your `TotoroUI` directory.
- The app will cache model loading for faster subsequent runs.

## 2. `requirements.txt`

streamlit
torch
numpy
Pillow
einops
diffusers
accelerate
xformers==0.0.28.post2
torchsde

*You may need to add other dependencies if your `nodes` or `totoro_extras` modules require them.*

## 3. Deployment Optimization Checklist

- ✅ **Model loading is cached** using `@st.cache_resource`.
- ✅ **No shell commands** in the main app; all setup is handled before running Streamlit.
- ✅ **All user inputs** are handled via Streamlit widgets in the sidebar.
- ✅ **Session state** is used to track the current image and seed.
- ✅ **Download button** uses an in-memory buffer for fast image download.
- ✅ **Editing** reuses the same seed for reproducibility.
- ✅ **No hardcoded Colab paths** or magic commands (`%cd`, `!pip`, etc.).

### If you need a sample `app.py` file as well (with all the code integrated), just ask! 

Let me know if you want any further tweaks or a zipped starter repo.
