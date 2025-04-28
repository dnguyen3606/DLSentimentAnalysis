import torch
torch.classes.__path__ = []
import streamlit as st
from label import predict_emotion
from generate import generate
from e2va import composite_va, classify_va
from utils.midi2wav import midi_to_wav
from utils.emopia import download_and_extract
import asyncio

try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

@st.cache_resource(show_spinner="Downloading setup files...")
def setup_files():
    with st.spinner("Downloading and preparing necessary files..."):
        download_and_extract("17dKUf33ZsDbHC5Z6rkQclge3ppDTVCMP")
        download_and_extract("19Seq18b2JNzOamEQMG1uarKjj27HJkHu")
    st.success("Model weights and dictionaries are ready!")
    return True

setup_files()

st.title("CS4644 - Text to Emotion to Music Generation")

user_input = st.text_input("Enter your text:")
if st.button("Generate Music"):
    emotion_probs = predict_emotion(text=user_input, returnProbs=True)
    composite_valence, composite_arousal = composite_va(emotion_probs=emotion_probs)
    emotion_class = classify_va(composite_valence=composite_valence, composite_arousal=composite_arousal)

    st.write(f"Text: {user_input}")
    st.write(f"Emotion Probabilities: {emotion_probs}")
    st.write(f"Composite Valence-Arousal Levels: {composite_valence}, {composite_arousal}")
    st.write(f"Emotion Class: Q{emotion_class}")

    midi_path = generate(emotion_tag=emotion_class)
    wav_path = midi_to_wav(midi_path=midi_path)

    st.write("Here’s your generated music based on the detected emotion!")
    st.audio(wav_path)