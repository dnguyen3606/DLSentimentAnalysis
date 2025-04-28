import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader 
from utils.models import LSTMModel, GoEmotionsDataset, BoWClassifier, BoWDataset
from transformers import AutoTokenizer
from sklearn.metrics import f1_score

df1 = pd.read_csv("data/go-emotions/goemotions_1.csv")
df2 = pd.read_csv("data/go-emotions/goemotions_2.csv")
df3 = pd.read_csv("data/go-emotions/goemotions_3.csv")

df = pd.concat([df1, df2, df3])
df.drop(columns=['example_very_unclear', 'rater_id', 'created_utc', 'link_id', 'parent_id', 'subreddit', 'author', 'id'], inplace=True)

class_names = df.columns[1:].tolist()

bow = True
if not(bow):
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', padding_side="left")
    model_pth = 'models/GoEmotionsLSTM_3-21.pth' 
    
    loaded_model = LSTMModel(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=300,
        hidden_size=128,
        num_classes=28
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(model_pth, map_location=device)
    loaded_model.load_state_dict(state_dict)
    loaded_model.to(device)
    loaded_model.eval()
    
    val_dataset = GoEmotionsDataset(df, df["text"].tolist())
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    thresholds = tune_thresholds(model=loaded_model, val_loader=val_loader, device=device)
else:
    model_pth = 'models/BOW.pth' 
    with open('models/count_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    
    X = vectorizer.transform(df['text'])
    y = df.drop(columns=['text']).values
    
    val_dataset = BoWDataset(X, y)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    input_dim = len(vectorizer.get_feature_names_out())
    output_dim = 28
    
    loaded_model = BoWClassifier(input_dim, 64, output_dim)
    
    
    loaded_model.load_state_dict(torch.load(model_pth))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded_model.to(device)
    loaded_model.eval()
    
    thresholds = tune_thresholds_bow(model=loaded_model, val_loader=val_loader, device=device)

    

def predict_emotion(text, model=loaded_model, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), thresholds=None, returnProbs=False, returnOne=False):
    inputs = tokenizer(
        text,
        max_length=150,
        padding='max_length',
        truncation=True,
        return_tensors='pt',
    )
    
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(inputs['input_ids'], inputs['attention_mask'])
        probs = torch.sigmoid(outputs).flatten()

    if thresholds is None:
        threshold_tensor = torch.full_like(probs, 0.5)
    else:
        threshold_tensor = torch.tensor(thresholds, dtype=probs.dtype, device=probs.device)

    preds = (probs >= threshold_tensor).int().cpu().numpy()

    if returnProbs:  # showProbs displays the probability of each class.
        emotion_probs = {class_names[i]: probs[i].item() for i in range(len(class_names))}
        emotion_probs = dict(sorted(emotion_probs.items(), key=lambda item: item[1], reverse=True))
        return emotion_probs

    if returnOne:  # returnOne returns only the highest probability class, ignoring thresholds.
        return class_names[probs.argmax().item()]

    return [class_names[i] for i, val in enumerate(preds) if val == 1]

def tune_thresholds(model=loaded_model, val_loader=val_loader, step=0.05, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    model.eval()
    all_outputs = []
    all_labels = []

    with torch.no_grad():
        for input_ids, attention_mask, labels in val_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            
            outputs = model(input_ids, attention_mask)
            probs = torch.sigmoid(outputs)
            
            all_outputs.append(probs.cpu())
            all_labels.append(labels.cpu())
    
    all_outputs = torch.cat(all_outputs)
    all_labels = torch.cat(all_labels)

    num_classes = all_labels.shape[1]
    thresholds = []

    for i in range(num_classes):
        best_threshold = 0.5
        best_f1 = 0.0
        for threshold in np.arange(0.0, 1.0 + step, step):
            preds = (all_outputs[:, i] >= threshold).int().numpy()
            true = all_labels[:, i].int().numpy()
            current_f1 = f1_score(true, preds, zero_division=0)
            if current_f1 > best_f1:
                best_f1 = current_f1
                best_threshold = threshold
        thresholds.append(best_threshold)

    # print("Optimal thresholds per class:")
    # for idx, thr in enumerate(thresholds):
    #     print(f"Label {idx}: {thr:.2f}")

    return thresholds

def predict_emotion_bow(text, model, vectorizer, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), thresholds=None, showProbs=False, returnOne=False):
    bow_vector = vectorizer.transform([text]).toarray()  
    bow_tensor = torch.tensor(bow_vector, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        outputs = model(bow_tensor)
        probs = torch.sigmoid(outputs).flatten()

    if thresholds is None:
        threshold_tensor = torch.full_like(probs, 0.5)
    else:
        threshold_tensor = torch.tensor(thresholds, dtype=probs.dtype, device=probs.device)

    preds = (probs >= threshold_tensor).cpu().int().numpy()

    if showProbs:  # Show probabilities of each class
        annotated_probs = {class_names[i]: probs[i].item() for i in range(len(class_names))}
        sorted_probs = dict(sorted(annotated_probs.items(), key=lambda item: item[1], reverse=True))
        print(f'Class probabilities for "{text}": {sorted_probs}')

    if returnOne:  # Return only the highest probability class, ignoring thresholds
        return class_names[probs.argmax().item()]

    return [class_names[i] for i, val in enumerate(preds) if val == 1]


def tune_thresholds_bow(model, val_loader, step=0.05, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    model.eval()
    all_outputs = []
    all_labels = []

    with torch.no_grad():
        for bow_vector, labels in val_loader:
            bow_vector = bow_vector.to(device)
            labels = labels.to(device)

            outputs = model(bow_vector)
            probs = torch.sigmoid(outputs)

            all_outputs.append(probs.cpu())
            all_labels.append(labels.cpu())
    
    all_outputs = torch.cat(all_outputs)
    all_labels = torch.cat(all_labels)

    num_classes = all_labels.shape[1]
    thresholds = []

    for i in range(num_classes):
        best_threshold = 0.5
        best_f1 = 0.0
        for threshold in np.arange(0.1, 1.0 + step, step):
            preds = (all_outputs[:, i] >= threshold).int().numpy()
            true = all_labels[:, i].int().numpy()
            current_f1 = f1_score(true, preds, zero_division=0)
            if current_f1 > best_f1:
                best_f1 = current_f1
                best_threshold = threshold
        thresholds.append(best_threshold)

    print("Optimal thresholds per class:")
    for idx, thr in enumerate(thresholds):
        print(f"Label {idx}: {thr:.2f}")
    
    return thresholds

