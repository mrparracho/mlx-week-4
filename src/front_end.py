import streamlit as st
import torch
from PIL import Image
import os
from train import select_device
from src.infer import load_model, load_model_self, load_model_qwen
from src.data.flickr_dataset import FlickrDataset
from torchvision import transforms
import random

# Pre-download dependencies at startup
@st.cache_resource
def download_dependencies():
    """Download all required dependencies at startup."""
    with st.spinner("Downloading dependencies..."):
        # Download NLTK data
        try:
            import nltk
            nltk.download('punkt', quiet=True)
            nltk.download('wordnet', quiet=True)
            st.success("✅ NLTK data downloaded")
        except Exception as e:
            st.warning(f"⚠️ NLTK download failed: {e}")
        
        # Pre-download CLIP model
        try:
            from transformers import CLIPVisionModel, CLIPProcessor
            CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True)
            CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            st.success("✅ CLIP model downloaded")
        except Exception as e:
            st.warning(f"⚠️ CLIP download failed: {e}")
        
        # Pre-download GPT-2 tokenizer
        try:
            from transformers import GPT2Tokenizer
            GPT2Tokenizer.from_pretrained('gpt2')
            st.success("✅ GPT-2 tokenizer downloaded")
        except Exception as e:
            st.warning(f"⚠️ GPT-2 download failed: {e}")
        
        # Pre-download Qwen model
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base", use_fast=True)
            AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B-Base")
            st.success("✅ Qwen model downloaded")
        except Exception as e:
            st.warning(f"⚠️ Qwen download failed: {e}")

# Download dependencies at startup
download_dependencies()

# Cache models to avoid reloading
@st.cache_resource
def load_cached_model(checkpoint_path, device, model_type, qwen_dir_param="Qwen/Qwen3-0.6B-Base"):
    """Load and cache a model."""
    if model_type == "cross":
        return load_model(checkpoint_path, device)
    elif model_type == "self":
        return load_model_self(checkpoint_path, device)
    elif model_type == "qwen":
        return load_model_qwen(checkpoint_path, device, qwen_dir=qwen_dir_param)

st.set_page_config(page_title="Image Captioning Comparison", layout="centered")
st.title("🖼️ Image Captioning: Cross-Attention vs Self-Attention vs Qwen")

# Sidebar for model and parameters
st.sidebar.header("Model & Generation Settings")

default_ckpt_cross = "checkpoints/checkpoint_epoch_10.pth"
default_ckpt_self = "checkpoints/best_model_eval_self_attn.pth"
default_ckpt_qwen = "checkpoints/best_model_qwen.pth"
default_qwen_dir = "Qwen/Qwen3-0.6B-Base"
ckpt_path_cross = st.sidebar.text_input("Cross-Attention checkpoint", value=default_ckpt_cross)
ckpt_path_self = st.sidebar.text_input("Self-Attention checkpoint", value=default_ckpt_self)
ckpt_path_qwen = st.sidebar.text_input("Qwen checkpoint", value=default_ckpt_qwen)
qwen_dir = st.sidebar.text_input("Qwen HuggingFace repo", value=default_qwen_dir)

device = st.sidebar.selectbox("Device", options=["cpu", "cuda", "mps", "auto"], index=0)
if device == "auto":
    device = select_device()

temperature = st.sidebar.slider("Temperature", min_value=0.1, max_value=2.0, value=0.8, step=0.05)
max_length = st.sidebar.slider("Max caption length", min_value=10, max_value=100, value=50, step=1)
num_beams = st.sidebar.slider("Num beams (beam search)", min_value=1, max_value=10, value=5, step=1)

default_data_dir = "data/flickr30k"
data_dir = st.sidebar.text_input("Test data directory", value=default_data_dir)
test_split = st.sidebar.selectbox("Test split", options=["test", "val", "train"], index=0)

# Option to upload or pick random image
tab1, tab2 = st.tabs(["Upload Image", "Random Test Image"])

selected_image = None
reference_caption = None

with tab1:
    st.header("Upload an Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        # Resize to half original size for display
        w, h = image.size
        image_disp = image.resize((w // 2, h // 2))
        st.image(image_disp, caption="Uploaded Image (50%)", use_container_width=False)
        selected_image = image

with tab2:
    st.header("Pick a Random Test Image")
    if st.button("Pick Random Image"):
        image_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        try:
            dummy_model = load_model(ckpt_path_cross, device)
            test_dataset = FlickrDataset(
                root_dir=data_dir,
                split=test_split,
                tokenizer=dummy_model.tokenizer,
                transform=image_transforms
            )
            if len(test_dataset) == 0:
                st.error("Test dataset is empty!")
            else:
                idx = random.randint(0, len(test_dataset) - 1)
                img, ref_caption, _ = test_dataset[idx]
                if isinstance(img, torch.Tensor):
                    img = transforms.ToPILImage()(img)
                w, h = img.size
                image_disp = img.resize((w // 2, h // 2))
                st.image(image_disp, caption="Random Test Image (50%)", use_container_width=False)
                selected_image = img
                reference_caption = ref_caption
                st.session_state["random_image"] = img
                st.session_state["random_caption"] = ref_caption
        except Exception as e:
            st.error(f"Error loading test dataset: {e}")
    elif "random_image" in st.session_state:
        img = st.session_state["random_image"]
        w, h = img.size
        image_disp = img.resize((w // 2, h // 2))
        st.image(image_disp, caption="Random Test Image (50%)", use_container_width=False)
        st.markdown(f"**Reference Caption:** {st.session_state['random_caption']}")
        selected_image = img
        reference_caption = st.session_state["random_caption"]

if selected_image is not None:
    if not os.path.exists(ckpt_path_cross):
        st.error(f"Cross-attention checkpoint not found: {ckpt_path_cross}")
    elif not os.path.exists(ckpt_path_self):
        st.error(f"Self-attention checkpoint not found: {ckpt_path_self}")
    elif not os.path.exists(ckpt_path_qwen):
        st.error(f"Qwen checkpoint not found: {ckpt_path_qwen}")
    else:
        with st.spinner("Loading models..."):
            model_cross = load_cached_model(ckpt_path_cross, device, "cross")
            st.success("✅ Cross-attention model loaded")
            
            model_self = load_cached_model(ckpt_path_self, device, "self")
            st.success("✅ Self-attention model loaded")
            
            generate_caption_qwen = load_cached_model(ckpt_path_qwen, device, "qwen", qwen_dir)
            st.success("✅ Qwen model loaded")
        
        with st.spinner("Generating captions..."):
            caption_cross = model_cross.generate_caption(
                selected_image,
                max_length=max_length,
                num_beams=num_beams,
                temperature=temperature
            )
            caption_self = model_self.generate_caption(
                selected_image,
                max_length=max_length,
                num_beams=num_beams,
                temperature=temperature
            )
            caption_qwen = generate_caption_qwen(
                selected_image,
                max_length=max_length
            )
        st.success("Captions generated!")
        # Display results in a table with three columns, each 1/3 width, no row label
        st.markdown("""
        <style>
        .caption-table {width: 100%; table-layout: fixed;}
        .caption-table th, .caption-table td {font-size: 1.1em !important; text-align: center; width: 33%;}
        </style>
        """, unsafe_allow_html=True)
        st.markdown(
            f"""
            <table class='caption-table'>
                <tr>
                    <th>Cross-Attention</th>
                    <th>Self-Attention</th>
                    <th>Qwen</th>
                </tr>
                <tr>
                    <td>{caption_cross}</td>
                    <td>{caption_self}</td>
                    <td>{caption_qwen}</td>
                </tr>
            </table>
            """,
            unsafe_allow_html=True
        )
        if reference_caption is not None:
            st.markdown(f"**Reference Caption:** {reference_caption}")
else:
    st.info("Please upload an image or pick a random test image to generate a caption.")
