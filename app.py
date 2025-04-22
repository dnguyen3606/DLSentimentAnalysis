import streamlit as st

from label import predict_emotion
from generate import generate
from e2va import composite_va, classify_va
from utils.midi2wav import midi_to_wav

st.title("CS4644 - Text to Emotion to Music Generation")

user_input = st.text_input("Enter your text:")
if st.button("Generate Music"):
    emotion_probs = predict_emotion(text=user_input)
    composite_valence, composite_arousal = composite_va(emotion_probs=emotion_probs)
    emotion_class = classify_va(composite_valence=composite_valence, composite_arousal=composite_arousal)

    print(f"Text: {user_input}")
    print(f"Emotion Probabilities: {emotion_probs}")
    print(f"Composite Valence-Arousal Levels: {composite_valence}, {composite_arousal}")
    print(f"Emotion Class: Q{emotion_class}")

    midi_path = generate(emotion_tag=emotion_class)
    wav_path = midi_to_wav(midi_path=midi_path)

    st.write("Here’s your generated music based on the detected emotion!")
    st.audio(wav_path)