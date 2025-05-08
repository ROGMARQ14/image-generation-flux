# Clone TotoroUI repo here
git clone -b totoro3 https://github.com/camenduru/ComfyUI TotoroUI

# Install all packags and libraries
import streamlit as st
import random
import torch
import numpy as np
from PIL import Image
import io
import os
import sys

def download_models_if_needed():
    models_path = os.path.join(os.getcwd(), "TotoroUI/models")
    os.makedirs(os.path.join(models_path, "unet"), exist_ok=True)
    os.makedirs(os.path.join(models_path, "vae"), exist_ok=True)
    os.makedirs(os.path.join(models_path, "clip"), exist_ok=True)
    
    files_to_download = {
        "unet/flux1-dev-fp8.safetensors": "https://huggingface.co/camenduru/FLUX.1-dev/resolve/main/flux1-dev-fp8.safetensors",
        "vae/ae.sft": "https://huggingface.co/camenduru/FLUX.1-dev/resolve/main/ae.sft",
        "clip/clip_l.safetensors": "https://huggingface.co/camenduru/FLUX.1-dev/resolve/main/clip_l.safetensors",
        "clip/t5xxl_fp8_e4m3fn.safetensors": "https://huggingface.co/camenduru/FLUX.1-dev/resolve/main/t5xxl_fp8_e4m3fn.safetensors"
    }
    
    for file_path, url in files_to_download.items():
        full_path = os.path.join(models_path, file_path)
        if not os.path.exists(full_path):
            st.info(f"Downloading {file_path}...")
            # Use wget or requests to download the file
            import requests
            response = requests.get(url, stream=True)
            with open(full_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

# Get the directory of THIS script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Add these paths (relative to your script)
sys.path.append(os.path.join(SCRIPT_DIR, "TotoroUI"))
sys.path.append(os.path.join(SCRIPT_DIR, "TotoroUI/custom_nodes"))

# Set page configuration
st.set_page_config(page_title="FLUX AI Image Generator", layout="wide")

# Core Components and Helper Functions
# First, let's define some helper functions that will be used throughout the application:

# Function to calculate height based on width and aspect ratio
def calculate_height(width, aspect_ratio):
    if aspect_ratio == "1:1":
        return width
    elif aspect_ratio == "2:3":
        return int(width * 3 / 2)
    elif aspect_ratio == "4:3":
        return int(width * 3 / 4)
    elif aspect_ratio == "16:9":
        return int(width * 9 / 16)
    elif aspect_ratio == "9:16":
        return int(width * 16 / 9)
    return width  # Default to square

def closestNumber(n, m):
    q = int(n / m)
    n1 = m * q
    if (n * m) > 0:
        n2 = m * (q + 1)
    else:
        n2 = m * (q - 1)
    if abs(n - n1) < abs(n - n2):
        return n1
    return n2

# Session State Management
# Streamlit reruns the entire script on each interaction, so we need to use session state to persist data between runs:

# Initialize session state
if 'generated_image' not in st.session_state:
    st.session_state.generated_image = None
if 'current_seed' not in st.session_state:
    st.session_state.current_seed = 0

# Building the User Interface
# Let's build the sidebar with all the required input elements:

# Add title
st.title("FLUX AI Image Generator")

# Sidebar UI components
with st.sidebar:
    st.header("Generation Parameters")
    
    # 1. Prompt field
    prompt = st.text_area("Prompt:", value="anime style", height=100)
    
    # 2. Aspect ratio field
    aspect_ratio = st.selectbox(
        "Aspect Ratio:",
        options=["1:1", "2:3", "4:3", "16:9", "9:16"],
        index=0
    )
    
    # 3. Dimensions fields
    width = st.number_input("Width:", min_value=256, max_value=2048, value=1024, step=64)
    height = calculate_height(width, aspect_ratio)
    st.text(f"Height: {height}")
    
    # 4. Seed field
    seed = st.number_input("Seed (0 for random):", min_value=0, value=0)
    
    # 5. Steps field
    steps = st.number_input("Steps:", min_value=1, max_value=100, value=20)
    
    # Generate button
    generate_button = st.button("Generate Image")

# Model Initialization and Caching
# To avoid reloading the models on every interaction, we'll use Streamlit's caching mechanism:

@st.cache_resource

def initialize_models():
    try:
        # Import modules
        import nodes
        from nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
        from totoro_extras import nodes_custom_sampler
        from totoro_extras import nodes_post_processing
        from totoro import model_management

        # Initialize nodes
        DualCLIPLoader = NODE_CLASS_MAPPINGS["DualCLIPLoader"]()
        UNETLoader = NODE_CLASS_MAPPINGS["UNETLoader"]()
        RandomNoise = nodes_custom_sampler.NODE_CLASS_MAPPINGS["RandomNoise"]()
        BasicGuider = nodes_custom_sampler.NODE_CLASS_MAPPINGS["BasicGuider"]()
        KSamplerSelect = nodes_custom_sampler.NODE_CLASS_MAPPINGS["KSamplerSelect"]()
        BasicScheduler = nodes_custom_sampler.NODE_CLASS_MAPPINGS["BasicScheduler"]()
        SamplerCustomAdvanced = nodes_custom_sampler.NODE_CLASS_MAPPINGS["SamplerCustomAdvanced"]()
        VAELoader = NODE_CLASS_MAPPINGS["VAELoader"]()
        VAEDecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
        VAEEncode = NODE_CLASS_MAPPINGS["VAEEncode"]()
        EmptyLatentImage = NODE_CLASS_MAPPINGS["EmptyLatentImage"]()
        ImageScaleToTotalPixels = nodes_post_processing.NODE_CLASS_MAPPINGS["ImageScaleToTotalPixels"]()
        LoadImage = nodes.LoadImage()

        # Load models
        with torch.inference_mode():
            clip = DualCLIPLoader.load_clip("t5xxl_fp8_e4m3fn.safetensors", "clip_l.safetensors", "flux")[0]
            unet = UNETLoader.load_unet("flux1-dev-fp8.safetensors", "fp8_e4m3fn")[0]
            vae = VAELoader.load_vae("ae.sft")[0]

        return {
            "nodes": nodes,
            "RandomNoise": RandomNoise,
            "BasicGuider": BasicGuider,
            "KSamplerSelect": KSamplerSelect,
            "BasicScheduler": BasicScheduler,
            "SamplerCustomAdvanced": SamplerCustomAdvanced,
            "VAEDecode": VAEDecode,
            "VAEEncode": VAEEncode,
            "EmptyLatentImage": EmptyLatentImage,
            "ImageScaleToTotalPixels": ImageScaleToTotalPixels,
            "LoadImage": LoadImage,
            "model_management": model_management,
            "clip": clip,
            "unet": unet,
            "vae": vae
        }
    except Exception as e:
        st.error(f"Error initializing models: {e}")
        return None

# Image Generation Function
# Now let's adapt the original image generation code to work with our Streamlit interface:

def generate_image(prompt, width, height, seed, steps):
    models = initialize_models()
    if models is None:
        return None, seed

    with torch.inference_mode():
        sampler_name = "euler"
        scheduler = "simple"

        if seed == 0:
            seed = random.randint(0, 18446744073709551615)

        # Extract models and nodes
        clip = models["clip"]
        unet = models["unet"]
        vae = models["vae"]
        RandomNoise = models["RandomNoise"]
        BasicGuider = models["BasicGuider"]
        KSamplerSelect = models["KSamplerSelect"]
        BasicScheduler = models["BasicScheduler"]
        SamplerCustomAdvanced = models["SamplerCustomAdvanced"]
        VAEDecode = models["VAEDecode"]
        VAEEncode = models["VAEEncode"]
        ImageScaleToTotalPixels = models["ImageScaleToTotalPixels"]
        LoadImage = models["LoadImage"]
        model_management = models["model_management"]

        # Encode the prompt
        cond, pooled = clip.encode_from_tokens(clip.tokenize(prompt), return_pooled=True)
        cond = [[cond, {"pooled_output": pooled}]]
        
        # Generate noise based on seed
        noise = RandomNoise.get_noise(seed)[0] 
        
        # Set up the guider, sampler, and scheduler
        guider = BasicGuider.get_guider(unet, cond)[0]
        sampler = KSamplerSelect.get_sampler(sampler_name)[0]
        sigmas = BasicScheduler.get_sigmas(unet, scheduler, steps, 0.75)[0]

        # Check if test.png exists, otherwise use an empty latent image
        if os.path.exists("test.png"):
            # Similar to the original code, use test.png as a starting point
            image = LoadImage.load_image("test.png")[0]
            latent_image = ImageScaleToTotalPixels.upscale(image, "lanczos", 1.0)[0]
            latent_image = VAEEncode.encode(vae, latent_image)[0]
        else:
            # If test.png doesn't exist, create an empty latent image
            latent_image = models["EmptyLatentImage"].generate(width, height, 1)[0]

        # Sample and decode
        sample, sample_denoised = SamplerCustomAdvanced.sample(noise, guider, sampler, sigmas, latent_image)
        model_management.soft_empty_cache()
        decoded = VAEDecode.decode(vae, sample)[0].detach()

        # Convert to PIL Image and resize if needed
        image_array = np.array(decoded*255, dtype=np.uint8)[0]
        generated_image = Image.fromarray(image_array)
        
        if generated_image.width != width or generated_image.height != height:
            generated_image = generated_image.resize((width, height), Image.LANCZOS)

        return generated_image, seed

# Layout and Image Display
# Let's set up the main content area with two columns - one for displaying the generated image and another for the editing section:

# Main content area
col1, col2 = st.columns([2, 1])

# Display area
with col1:
    if generate_button:
        with st.spinner('Generating image...'):
            # Generate the image
            generated_image, actual_seed = generate_image(prompt, width, height, seed, steps)
            
            if generated_image is not None:
                st.session_state.generated_image = generated_image
                st.session_state.current_seed = actual_seed
                
                # Display seed information if random seed was used
                if seed == 0:
                    st.sidebar.text(f"Generated with seed: {actual_seed}")
    
    # Display the generated image if it exists
    if st.session_state.generated_image is not None:
        st.image(st.session_state.generated_image, caption="Generated Image", use_column_width=True)
# Download Button Implementation
# We'll implement a download button to allow users to save their generated images:

# Download button
if st.session_state.generated_image is not None:
    buf = io.BytesIO()
    st.session_state.generated_image.save(buf, format="PNG")
    buf.seek(0)
    
    st.download_button(
        label="Download Image",
        data=buf,
        file_name="generated_image.png",
        mime="image/png"
    )

# Image Editing Section
# Finally, let's add the editing section that allows users to modify the generated image with text prompts while maintaining the same seed:

# Edit field
with col2:
    if st.session_state.generated_image is not None:
        st.header("Edit Image")
        edit_prompt = st.text_area(
            "Describe changes to apply:",
            height=100,
            help="Describe changes to apply to the current image."
        )
        
        apply_changes = st.button("Apply Changes")
        
        if apply_changes and edit_prompt:
            with st.spinner('Applying changes...'):
                # Create a new prompt combining the original and edit prompts
                combined_prompt = f"{prompt}, {edit_prompt}"
                
                # Generate the edited image using the same seed
                edited_image, _ = generate_image(
                    combined_prompt, 
                    width, 
                    height, 
                    st.session_state.current_seed, 
                    steps
                )
                
                if edited_image is not None:
                    st.session_state.generated_image = edited_image
