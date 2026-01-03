"""
RoBERTa Sentiment Scoring for Full and Annotated Samples
"""

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
from tqdm import tqdm
from datetime import datetime
import os
import warnings

warnings.filterwarnings('ignore')

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

# FULL_SAMPLE = ""
ANNOTATED_SAMPLE = ""
OUTPUT_DIR = ""

BATCH_SIZE = 32


def load_model():
    print("Loading RoBERTa model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")
    model.to(device)
    model.eval()
    
    return tokenizer, model, device


def get_roberta_scores_batch(texts, tokenizer, model, device):
    encoded = tokenizer(
        texts,
        return_tensors='pt',
        truncation=True,
        padding=True,
        max_length=512
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}
    
    with torch.no_grad():
        output = model(**encoded)
    
    scores = F.softmax(output.logits, dim=1).cpu().numpy()
    
    results = []
    for i in range(len(texts)):
        neg, neu, pos = scores[i][0], scores[i][1], scores[i][2]
        valence = pos + neg
        results.append({
            'roberta_negative': neg,
            'roberta_neutral': neu,
            'roberta_positive': pos,
            'roberta_valence': valence
        })
    
    return results


def process_dataframe(df, tokenizer, model, device):
    messages = df['message'].fillna('').astype(str).tolist()
    
    all_results = []
    
    for i in tqdm(range(0, len(messages), BATCH_SIZE), desc="Processing"):
        batch = messages[i:i+BATCH_SIZE]
        batch_results = get_roberta_scores_batch(batch, tokenizer, model, device)
        all_results.extend(batch_results)
    
    results_df = pd.DataFrame(all_results)
    
    for col in results_df.columns:
        df[col] = results_df[col].values
    
    return df


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("="*60)
    print("ROBERTA SENTIMENT SCORING")
    print("="*60)
    
    tokenizer, model, device = load_model()
    
    print("\n" + "="*60)
    # print("PROCESSING FULL SAMPLE")
    # print("="*60)
    
    # df_full = pd.read_csv(FULL_SAMPLE)
    # print(f"Loaded {len(df_full)} messages")
    
    # cols_to_drop = [c for c in df_full.columns if c.startswith('roberta_')]
    # if cols_to_drop:
    #     df_full = df_full.drop(columns=cols_to_drop)
    #     print(f"Dropped existing RoBERTa columns: {cols_to_drop}")
    
    # df_full = process_dataframe(df_full, tokenizer, model, device)
    
    # full_output = f"{OUTPUT_DIR}/full_sample_with_roberta_{timestamp}.csv"
    # df_full.to_csv(full_output, index=False)
    # print(f"Saved: {full_output}")
    
    print("\n" + "="*60)
    print("PROCESSING ANNOTATED SAMPLE")
    print("="*60)
    
    df_ann = pd.read_csv(ANNOTATED_SAMPLE)
    print(f"Loaded {len(df_ann)} messages")
    
    cols_to_drop = [c for c in df_ann.columns if c.startswith('roberta_')]
    if cols_to_drop:
        df_ann = df_ann.drop(columns=cols_to_drop)
        print(f"Dropped existing RoBERTa columns: {cols_to_drop}")
    
    df_ann = process_dataframe(df_ann, tokenizer, model, device)
    
    ann_output = f"{OUTPUT_DIR}/cbt_control_with_roberta_{timestamp}.csv"
    df_ann.to_csv(ann_output, index=False)
    print(f"Saved: {ann_output}")
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    # print("\nFull Sample RoBERTa Statistics:")
    # print(f"  Negative: M={df_full['roberta_negative'].mean():.4f}, SD={df_full['roberta_negative'].std():.4f}")
    # print(f"  Neutral:  M={df_full['roberta_neutral'].mean():.4f}, SD={df_full['roberta_neutral'].std():.4f}")
    # print(f"  Positive: M={df_full['roberta_positive'].mean():.4f}, SD={df_full['roberta_positive'].std():.4f}")
    # print(f"  Valence:  M={df_full['roberta_valence'].mean():.4f}, SD={df_full['roberta_valence'].std():.4f}")
    
    print("\ Sample RoBERTa Statistics:")
    print(f"  Negative: M={df_ann['roberta_negative'].mean():.4f}, SD={df_ann['roberta_negative'].std():.4f}")
    print(f"  Neutral:  M={df_ann['roberta_neutral'].mean():.4f}, SD={df_ann['roberta_neutral'].std():.4f}")
    print(f"  Positive: M={df_ann['roberta_positive'].mean():.4f}, SD={df_ann['roberta_positive'].std():.4f}")
    print(f"  Valence:  M={df_ann['roberta_valence'].mean():.4f}, SD={df_ann['roberta_valence'].std():.4f}")
    
    print("\n" + "="*60)
    print("FILES SAVED")
    print("="*60)
    print(f"  {full_output}")
    print(f"  {ann_output}")


if __name__ == "__main__":
    main()

