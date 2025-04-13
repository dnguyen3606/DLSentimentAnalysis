emotion_va_mappings = {
    'admiration':      (0.8, 0.6),  
    'amusement':       (0.9, 0.8),  
    'anger':           (0.2, 0.9),   
    'annoyance':       (0.3, 0.4),   
    'approval':        (0.8, 0.5),   
    'caring':          (0.85, 0.4),
    'confusion':       (0.4, 0.5),   
    'curiosity':       (0.7, 0.7),  
    'desire':          (0.8, 0.75),  
    'disappointment':  (0.3, 0.4),   
    'disapproval':     (0.25, 0.45), 
    'disgust':         (0.1, 0.7),  
    'embarrassment':   (0.3, 0.6),  
    'excitement':      (0.9, 0.9),   
    'fear':            (0.2, 0.8),  
    'gratitude':       (0.85, 0.4), 
    'grief':           (0.1, 0.3),  
    'joy':             (0.9, 0.8),   
    'love':            (0.95, 0.7),  
    'nervousness':     (0.3, 0.8),   
    'optimism':        (0.85, 0.6),  
    'pride':           (0.9, 0.7),   
    'realization':     (0.6, 0.5),   
    'relief':          (0.9, 0.4),   
    'remorse':         (0.2, 0.5), 
    'sadness':         (0.2, 0.3),   
    'surprise':        (0.5, 0.8),   
    'neutral':         (0.5, 0.5),   
}

def composite_va(emotion_probs, mapping=emotion_va_mappings):
    weighted_valence = 0.0
    weighted_arousal = 0.0
    total_prob = 0.0

    for emotion, prob in emotion_probs.items():
        if emotion in mapping:
            v_weight, a_weight = mapping[emotion]
            weighted_valence += prob * v_weight
            weighted_arousal += prob * a_weight
            total_prob += prob
        else:
            print(f"Warning: '{emotion}' not found in mappings. What did you do???")

    if total_prob > 0:
        composite_valence = weighted_valence / total_prob
        composite_arousal = weighted_arousal / total_prob
    else: # shouldnt happen
        composite_valence, composite_arousal = 0.5, 0.5

    return composite_valence, composite_arousal

def classify_va(composite_valence, composite_arousal):
    if composite_valence >= 0.5 and composite_arousal >= 0.5:
        return 1
    elif composite_valence < 0.5 and composite_arousal >= 0.5:
        return 2
    elif composite_valence < 0.5 and composite_arousal < 0.5:
        return 3
    elif composite_valence >= 0.5 and composite_arousal < 0.5:
        return 4
    else:
        print("something terrible happened")