#!/usr/bin/env python3
"""
Conversation Replay with Alternative LLMs

Design: Replay annotated user conversations with different LLMs to compare
emotional attunement across models.

Method:
1. Extract user messages from annotated sample in conversation order
2. For each conversation, replay turn-by-turn with target LLM
3. LLM builds its own conversation history (responds to its OWN previous response)
4. Compute sentiment on generated responses
5. Save replayed dataset for cross-lag analysis

Author: Marx
Date: 2025-12
"""

import pandas as pd
import numpy as np
import requests
from datetime import datetime
import os
import sys
import time
import warnings

warnings.filterwarnings('ignore')

# Force unbuffered output for real-time logging
sys.stdout.reconfigure(line_buffering=True)

# Ollama API endpoint
OLLAMA_API = "http://localhost:11434/api/chat"

# ============================================================================
# CONFIGURATION
# ============================================================================

INPUT_FILE = ""
OUTPUT_DIR = ""
RESULTS_DIR = ""
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')

# Model to use (available on your system)
# MODEL_NAME = "llama3.2:latest"  # 3B local model - fast and clean
MODEL_NAME = "deepseek-r1:latest"
# System prompt for consistent behavior
SYSTEM_PROMPT = """You are a helpful, empathetic AI assistant engaged in a conversation. 
Respond naturally and supportively to the user's messages. 
Keep responses concise but thoughtful (2-4 sentences typically, longer if needed for complex topics)."""

# ============================================================================
# DATA EXTRACTION
# ============================================================================

def load_and_extract_conversations(filepath):
    """
    Load annotated sample and extract user messages per conversation.
    Returns dict: {conversation_id: [list of user messages in order]}
    """
    print(f"Loading: {filepath}")
    df = pd.read_csv(filepath)
    
    print(f"  Total messages: {len(df)}")
    print(f"  Conversations: {df['conversation_id'].nunique()}")
    
    conversations = {}
    conversation_metadata = {}
    
    for conv_id, group in df.groupby('conversation_id'):
        group = group.sort_values('timestamp')
        
        # Extract user messages only
        user_msgs = group[group['user_or_ai'] == 'user']['message'].tolist()
        
        if len(user_msgs) > 0:
            conversations[conv_id] = user_msgs
            conversation_metadata[conv_id] = {
                'participant_id': group['participant_id'].iloc[0],
                'session_date': group['session_date'].iloc[0] if 'session_date' in group.columns else None,
                'n_turns': len(user_msgs)
            }
    
    print(f"  Extracted {len(conversations)} conversations with user messages")
    total_turns = sum(len(msgs) for msgs in conversations.values())
    print(f"  Total user turns to replay: {total_turns}")
    
    return conversations, conversation_metadata


# ============================================================================
# LLM INTERACTION
# ============================================================================

def call_llm(messages, model=MODEL_NAME):
    """
    Call Ollama LLM with conversation history via direct HTTP API.
    
    Args:
        messages: List of {"role": "user"/"assistant", "content": "..."}
        model: Model name
    
    Returns:
        Generated response text
    """
    try:
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "think": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "num_predict": 200,
            }
        }
        
        response = requests.post(OLLAMA_API, json=payload, timeout=120)
        response.raise_for_status()
        
        result = response.json()
        return result['message']['content'].strip()
    except Exception as e:
        print(f"    Error calling LLM: {e}")
        return f"[Error: {str(e)[:50]}]"


def replay_conversation(user_messages, model=MODEL_NAME, conv_idx=0, conv_id=""):
    """
    Replay a conversation turn-by-turn with the target LLM.
    
    The LLM responds to its OWN previous responses, building natural history.
    
    Args:
        user_messages: List of user message strings in order
        model: LLM model name
        conv_idx: Conversation index for logging
        conv_id: Conversation ID for logging
    
    Returns:
        List of dicts: [{"turn": 1, "user": "...", "ai": "..."}, ...]
    """
    results = []
    history = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    print(f"\n[Conv {conv_idx+1}] {conv_id[:30]}... ({len(user_messages)} turns)", flush=True)
    
    for turn_idx, user_msg in enumerate(user_messages, 1):
        # Add user message to history
        history.append({"role": "user", "content": user_msg})
        
        # Log user message
        user_preview = user_msg[:60].replace('\n', ' ')
        print(f"  T{turn_idx} USER: {user_preview}...", flush=True)
        
        # Generate response
        ai_response = call_llm(history, model)
        
        # Add AI response to history for next turn
        history.append({"role": "assistant", "content": ai_response})
        
        # Log AI response
        ai_preview = ai_response[:60].replace('\n', ' ')
        print(f"  T{turn_idx} AI:   {ai_preview}...", flush=True)
        
        # Store result
        results.append({
            "turn": turn_idx,
            "user_message": user_msg,
            "ai_response": ai_response
        })
    
    return results



# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("=" * 70)
    print("CONVERSATION REPLAY WITH ALTERNATIVE LLM")
    print("=" * 70)
    print(f"\nModel: {MODEL_NAME}")
    print(f"Timestamp: {TIMESTAMP}")
    
    # Create output directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Load and extract conversations
    conversations, metadata = load_and_extract_conversations(INPUT_FILE)
    
    # Verify LLM connection
    print(f"\nTesting LLM connection ({MODEL_NAME})...")
    test_response = call_llm([
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, can you hear me?"}
    ])
    print(f"  Test response: {test_response[:100]}...")
    
    # Replay all conversations
    print(f"\n{'='*70}")
    print("REPLAYING CONVERSATIONS")
    print(f"{'='*70}")
    
    all_results = []
    total_convs = len(conversations)
    
    for idx, (conv_id, user_msgs) in enumerate(conversations.items()):
        # Replay conversation with detailed logging
        turns = replay_conversation(user_msgs, MODEL_NAME, conv_idx=idx, conv_id=conv_id)
        
        # Store results
        for turn_data in turns:
            # User message row
            all_results.append({
                'conversation_id': conv_id,
                'participant_id': metadata[conv_id]['participant_id'],
                'turn': turn_data['turn'],
                'user_or_ai': 'user',
                'message': turn_data['user_message'],
                'model': 'original_user',
                'timestamp': f"{idx:04d}_{turn_data['turn']:02d}_user"
            })
            
            # AI response row
            all_results.append({
                'conversation_id': conv_id,
                'participant_id': metadata[conv_id]['participant_id'],
                'turn': turn_data['turn'],
                'user_or_ai': 'ai',
                'message': turn_data['ai_response'],
                'model': MODEL_NAME,
                'timestamp': f"{idx:04d}_{turn_data['turn']:02d}_ai"
            })
        
        # Progress update
        print(f"  [✓] Conv {idx + 1}/{total_convs} complete ({len(turns)} turns)", flush=True)
    
    # Create dataframe
    df = pd.DataFrame(all_results)
    
    # Add sentiment scores
    df = add_sentiment_scores(df)
    
    # Save replayed dataset
    output_file = os.path.join(OUTPUT_DIR, f"replay_{MODEL_NAME.replace(':', '_')}_{TIMESTAMP}.csv")
    df.to_csv(output_file, index=False)
    print(f"\nSaved: {output_file}")
    
    # Generate summary statistics
    print(f"\n{'='*70}")
    print("SUMMARY STATISTICS")
    print(f"{'='*70}")
    
    print(f"\nDataset: {len(df)} messages")
    print(f"  Conversations: {df['conversation_id'].nunique()}")
    print(f"  User messages: {len(df[df['user_or_ai'] == 'user'])}")
    print(f"  AI responses: {len(df[df['user_or_ai'] == 'ai'])}")
    
    print("\nSentiment Summary (AI responses only):")
    ai_df = df[df['user_or_ai'] == 'ai']
    print(f"  VADER compound: M={ai_df['vader_compound'].mean():.4f}, SD={ai_df['vader_compound'].std():.4f}")
    print(f"  NRC Valence:    M={ai_df['nrc_valence'].mean():.4f}, SD={ai_df['nrc_valence'].std():.4f}")
    print(f"  NRC Arousal:    M={ai_df['nrc_arousal'].mean():.4f}, SD={ai_df['nrc_arousal'].std():.4f}")
    
    # Save summary
    summary_file = os.path.join(RESULTS_DIR, f"replay_summary_{TIMESTAMP}.txt")
    with open(summary_file, 'w') as f:
        f.write("CONVERSATION REPLAY SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Model: {MODEL_NAME}\n")
        f.write(f"Timestamp: {TIMESTAMP}\n")
        f.write(f"Input: {INPUT_FILE}\n")
        f.write(f"Output: {output_file}\n\n")
        f.write(f"Total messages: {len(df)}\n")
        f.write(f"Conversations: {df['conversation_id'].nunique()}\n")
        f.write(f"User messages: {len(df[df['user_or_ai'] == 'user'])}\n")
        f.write(f"AI responses: {len(df[df['user_or_ai'] == 'ai'])}\n\n")
        f.write("AI Response Sentiment:\n")
        f.write(f"  VADER: M={ai_df['vader_compound'].mean():.4f}, SD={ai_df['vader_compound'].std():.4f}\n")
        f.write(f"  NRC Valence: M={ai_df['nrc_valence'].mean():.4f}, SD={ai_df['nrc_valence'].std():.4f}\n")
        f.write(f"  NRC Arousal: M={ai_df['nrc_arousal'].mean():.4f}, SD={ai_df['nrc_arousal'].std():.4f}\n")
    
    print(f"\nSaved: {summary_file}")
    
    return df


if __name__ == "__main__":
    df = main()

