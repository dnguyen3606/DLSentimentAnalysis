from label import predict_emotion
from generate import generate
from e2va import composite_va, classify_va
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("text", type=str, help="The text input to classify and generate music on.")
    args = parser.parse_args()

    emotion_probs = predict_emotion(args.text, returnProbs=True)
    composite_valence, composite_arousal = composite_va(emotion_probs)
    emotion_class = classify_va(composite_valence, composite_arousal)

    print(f"Text: {args.text}")
    print(f"Emotion Probabilities: {emotion_probs}")
    print(f"Composite Valence-Arousal Levels: {composite_valence}, {composite_arousal}")
    print(f"Emotion Class: Q{emotion_class}")

    generate(emotion_tag=emotion_class)