import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

st.set_page_config(page_title="Deep Learning Sentiment", page_icon="🚀")
st.title("🚀 GPU-Accelerated Sentiment Analysis")

# Check if CUDA (GPU) is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with st.sidebar:
    st.header("Hardware Info")
    if device.type == "cuda":
        st.success("🟢 Running on NVIDIA GPU (RTX 3050)")
        st.write(f"Device Name: {torch.cuda.get_device_name(0)}")
    else:
        st.error("🔴 Running on CPU")

def load_model():
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    # Safely move model to GPU
    model.to(device)
    model.eval() # Set to evaluation mode
    return tokenizer, model

st.write("This app uses **Hugging Face Transformers** and **PyTorch** with a custom GPU inference loop. Try pasting complex reviews or sarcastic comments!")

user_input = st.text_area("Enter movie review or text:", height=150)

if st.button("Predict using Deep Learning"):
    if user_input.strip() == "":
        st.warning("Please enter some text first.")
    else:
        with st.spinner("Loading Deep Learning model to GPU and analyzing context..."):
            try:
                tokenizer, model = load_model()
                
                # Custom Inference Loop
                inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True).to(device)
                
                with torch.no_grad():
                    logits = model(**inputs).logits
                
                # Get prediction
                predicted_class_id = logits.argmax().item()
                label = model.config.id2label[predicted_class_id]
                
                if label == "POSITIVE":
                    st.success(f"Sentiment: **{label}** 🥳")
                else:
                    st.error(f"Sentiment: **{label}** 😡")
            except Exception as e:
                st.error(f"Inference crashed: {e}")
