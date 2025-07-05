import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import io

# Custom CSS for a unique look
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #232526 0%, #414345 100%);
    color: #f5f6fa;
}
.stApp {
    background: linear-gradient(135deg, #232526 0%, #414345 100%);
}
header, .stTitle, .stTextInput, .stButton, .stDownloadButton {
    font-family: 'Segoe UI', 'Roboto', Arial, sans-serif;
}
.stTitle h1 {
    color: #f5f6fa;
    font-size: 2.5em;
    letter-spacing: 2px;
    text-shadow: 2px 2px 8px #00000055;
}
.stTextInput>div>div>input {
    border-radius: 12px;
    border: 2px solid #4F8BF9;
    padding: 0.7em;
    font-size: 1.2em;
    background: #232526;
    color: #f5f6fa;
}
.stButton>button, .stDownloadButton>button {
    background: linear-gradient(90deg, #4F8BF9 0%, #38b6ff 100%);
    color: white;
    font-weight: bold;
    border-radius: 12px;
    padding: 0.7em 2.5em;
    font-size: 1.2em;
    margin-top: 1em;
    box-shadow: 0 2px 8px #00000022;
    border: none;
}
.stSpinner {
    color: #4F8BF9;
}
.stImage>img {
    border-radius: 16px;
    box-shadow: 0 4px 24px #00000044;
    margin-top: 1.5em;
}
</style>
""", unsafe_allow_html=True)

st.set_page_config(page_title="Text-to-Image Generator", page_icon="🎨", layout="centered")
st.title("🎨 Text-to-Image Generator")

prompt = st.text_input("Enter your image prompt:", "A futuristic cityscape at sunset")
generate = st.button("Generate Image")

if generate and prompt:
    with st.spinner("Generating image, please wait..."):
        pipe = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        if torch.cuda.is_available():
            pipe = pipe.to("cuda")
        else:
            pipe = pipe.to("cpu")
        image = pipe(prompt).images[0]
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        st.image(buf.getvalue(), caption="Generated Image", use_container_width=True)
        st.download_button(
            label="Download Image",
            data=buf.getvalue(),
            file_name="generated_image.png",
            mime="image/png"
        )
        st.success("Image generated successfully!")
