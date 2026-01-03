#!/usr/bin/env python3
"""
Immediate Response Analysis for Emotional Attunement

Analyzes immediate and cross-lagged effects:
- IMMEDIATE: User_t → AI_t (AI responds to CURRENT user)
- CROSS-LAG: User_{t-1} → AI_t (AI attunes to PREVIOUS user)

Model Structure:
- AI immediate:   AI_{i} ~ User_{i} + AI_{i-1}
- AI cross-lag:   AI_{i} ~ User_{i-1} + AI_{i-1}
- User immediate: User_{i+1} ~ AI_{i} + User_{i}
- User cross-lag: User_{i} ~ AI_{i-1} + User_{i-1}

Usage:
    python immediate_response_analysis.py <file_path> <analysis_name>
    python immediate_response_analysis.py --batch  # runs predefined batch
    python immediate_response_analysis.py  # runs default samples
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats
import warnings
from datetime import datetime
import os
import sys
import argparse

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = ""
OUTPUT_DIR = ""
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')

# Default samples
DEFAULT_SAMPLES = [
    (f"{BASE_DIR}/full_sample_with_roberta_20251218_090606.csv", "Full_N447"),
    (f"{BASE_DIR}/annotated_sample_with_roberta_20251218_090606.csv", "Annotated_N73")
]

# Batch mode: all model comparison files
LLM_DIR = f"{BASE_DIR}/llm_replay"
BATCH_SAMPLES = [
    (f"{BASE_DIR}/annotated_sample_with_roberta_20251218_090606.csv", "ChatGPT_N73"),
    (f"{LLM_DIR}/cbt_control_with_roberta_20251222_204311.csv", "CBT_Human"),
    (f"{LLM_DIR}/llama3_with_roberta_20251222_135146.csv", "Llama3"),
    (f"{LLM_DIR}/deepseek_r1_with_roberta_20251222_200230.csv", "DeepSeek_R1"),
    (f"{LLM_DIR}/ministral_3_with_roberta_20251222_202813.csv", "Ministral_3"),
    (f"{LLM_DIR}/qwen3_8b_with_roberta_20251222_143342.csv", "Qwen3_8B"),
    (f"{BASE_DIR}/full_sample_with_roberta_20251218_090606.csv", "Full_N447"),
]

MEASURES = {
    'VADER': 'vader_compound_01',
    'NRC_Valence': 'nrc_valence',
    'NRC_Arousal': 'nrc_arousal',
    'NRC_Dominance': 'nrc_dominance',
    'RoBERTa_Sum': 'roberta_valence',
    'RoBERTa_Positive': 'roberta_positive',
    'RoBERTa_Negative': 'roberta_negative'
}


# ============================================================================
# DATA PREPARATION
# ============================================================================

def load_data(filepath):
    df = pd.read_csv(filepath)
    df['vader_compound_01'] = (df['vader_compound'] + 1) / 2
    return df


def create_sequential_data(df, measure_col):
    """
    Create data for immediate response analysis.
    
    For each exchange i:
    - user_i: User's message in exchange i
    - ai_i: AI's response in exchange i (responds to user_i)
    - user_i_prev: User's previous message (for autoregression)
    - ai_i_prev: AI's previous response (for autoregression)
    """
    exchanges = []
    
    for conv_id, conv_df in df.groupby('conversation_id'):
        conv_df = conv_df.sort_values('timestamp')
        participant_id = conv_df['participant_id'].iloc[0]
        
        user_msgs = conv_df[conv_df['user_or_ai'] == 'user'].reset_index(drop=True)
        ai_msgs = conv_df[conv_df['user_or_ai'] == 'ai'].reset_index(drop=True)
        
        n = min(len(user_msgs), len(ai_msgs))
        
        for i in range(n):
            exchanges.append({
                'conversation_id': conv_id,
                'participant_id': participant_id,
                'exchange_idx': i + 1,
                'user_current': user_msgs.iloc[i][measure_col],
                'ai_current': ai_msgs.iloc[i][measure_col]
            })
    
    ex_df = pd.DataFrame(exchanges)
    
    # Create lags (within conversation)
    ex_df['user_prev'] = ex_df.groupby('conversation_id')['user_current'].shift(1)
    ex_df['ai_prev'] = ex_df.groupby('conversation_id')['ai_current'].shift(1)
    
    # Create leads for user (next user message)
    ex_df['user_next'] = ex_df.groupby('conversation_id')['user_current'].shift(-1)
    
    return ex_df


def apply_centering(df):
    conv_means = df.groupby('conversation_id').agg({
        'user_current': 'mean',
        'ai_current': 'mean'
    }).rename(columns={'user_current': 'user_mean', 'ai_current': 'ai_mean'})
    
    df = df.merge(conv_means, on='conversation_id', how='left')
    
    df['user_c'] = df['user_current'] - df['user_mean']
    df['ai_c'] = df['ai_current'] - df['ai_mean']
    df['user_prev_c'] = df.groupby('conversation_id')['user_c'].shift(1)
    df['ai_prev_c'] = df.groupby('conversation_id')['ai_c'].shift(1)
    df['user_next_c'] = df.groupby('conversation_id')['user_c'].shift(-1)
    
    return df


def standardize(df, cols):
    df = df.copy()
    for col in cols:
        if col in df.columns and df[col].std() > 0:
            df[f'{col}_z'] = (df[col] - df[col].mean()) / df[col].std()
    return df


# ============================================================================
# MODELS
# ============================================================================

def run_model(df, formula, group_col):
    """
    Run mixed-effects model with proper NA handling.
    Only drop rows where formula variables are NaN.
    """
    try:
        parts = formula.replace(' ', '').split('~')
        dv = parts[0]
        ivs = [v.strip() for v in parts[1].split('+')]
        required_cols = [dv] + ivs
        
        clean_df = df.dropna(subset=required_cols)
        
        if len(clean_df) < 50:
            return None
        model = smf.mixedlm(
            formula,
            data=clean_df,
            groups=clean_df[group_col]
        ).fit(method='powell', disp=False)
        
        return model
    except Exception as e:
        print(f"      Model error: {e}")
        return None


def analyze_measure(df, measure_name, measure_col):
    print(f"\n  {measure_name}:")
    
    ex_df = create_sequential_data(df, measure_col)
    ex_df = apply_centering(ex_df)
    
    cols = ['user_current', 'ai_current', 'user_prev', 'ai_prev', 'user_next',
            'user_c', 'ai_c', 'user_prev_c', 'ai_prev_c', 'user_next_c']
    ex_df = standardize(ex_df, cols)
    
    required = ['user_c_z', 'ai_c_z', 'user_prev_c_z', 'ai_prev_c_z']
    missing = [c for c in required if c not in ex_df.columns]
    if missing:
        print(f"    Missing columns: {missing}")
        return None
    
    ex_clean = ex_df.dropna(subset=required).copy()
    
    if len(ex_clean) < 50:
        print(f"    Insufficient data: {len(ex_clean)}")
        return None
    
    print(f"    Data: {len(ex_clean)} exchanges")
    
    results = {
        'measure': measure_name,
        'n_exchanges': len(ex_clean),
        'n_conversations': ex_clean['conversation_id'].nunique()
    }
    
    # Model 1: AI IMMEDIATE response (AI ~ User_current)
    print("    AI immediate (AI ~ User_current + AI_prev)...")
    m_ai = run_model(ex_clean, "ai_c_z ~ user_c_z + ai_prev_c_z", 'conversation_id')
    
    if m_ai:
        results['ai_imm_user_beta'] = m_ai.fe_params['user_c_z']
        results['ai_imm_user_se'] = m_ai.bse['user_c_z']
        results['ai_imm_user_p'] = m_ai.pvalues['user_c_z']
        results['ai_ar_beta'] = m_ai.fe_params['ai_prev_c_z']
        results['ai_ar_p'] = m_ai.pvalues['ai_prev_c_z']
        print(f"      U→AI(imm): β={results['ai_imm_user_beta']:.4f}, p={results['ai_imm_user_p']:.4f}")
    
    # Model 2: User IMMEDIATE response (User_next ~ AI_current)
    print("    User immediate (User_next ~ AI_current + User_current)...")
    if 'user_next_c_z' in ex_clean.columns:
        ex_user = ex_clean.dropna(subset=['user_next_c_z'])
        if len(ex_user) > 50:
            m_user = run_model(ex_user, "user_next_c_z ~ ai_c_z + user_c_z", 'conversation_id')
            if m_user:
                results['user_imm_ai_beta'] = m_user.fe_params['ai_c_z']
                results['user_imm_ai_se'] = m_user.bse['ai_c_z']
                results['user_imm_ai_p'] = m_user.pvalues['ai_c_z']
                results['user_ar_beta'] = m_user.fe_params['user_c_z']
                results['user_ar_p'] = m_user.pvalues['user_c_z']
                print(f"      AI→U(imm): β={results['user_imm_ai_beta']:.4f}, p={results['user_imm_ai_p']:.4f}")
    
    # Model 3: AI CROSS-LAG (AI ~ User_prev)
    print("    AI cross-lag (AI ~ User_prev + AI_prev)...")
    m_ai_lag = run_model(ex_clean, "ai_c_z ~ user_prev_c_z + ai_prev_c_z", 'conversation_id')
    if m_ai_lag:
        results['ai_lag_user_beta'] = m_ai_lag.fe_params['user_prev_c_z']
        results['ai_lag_user_se'] = m_ai_lag.bse['user_prev_c_z']
        results['ai_lag_user_p'] = m_ai_lag.pvalues['user_prev_c_z']
        print(f"      U→AI(lag): β={results['ai_lag_user_beta']:.4f}, p={results['ai_lag_user_p']:.4f}")
    
    # Model 4: User CROSS-LAG (User ~ AI_prev)
    print("    User cross-lag (User ~ AI_prev + User_prev)...")
    m_user_lag = run_model(ex_clean, "user_c_z ~ ai_prev_c_z + user_prev_c_z", 'conversation_id')
    if m_user_lag:
        results['user_lag_ai_beta'] = m_user_lag.fe_params['ai_prev_c_z']
        results['user_lag_ai_se'] = m_user_lag.bse['ai_prev_c_z']
        results['user_lag_ai_p'] = m_user_lag.pvalues['ai_prev_c_z']
        print(f"      AI→U(lag): β={results['user_lag_ai_beta']:.4f}, p={results['user_lag_ai_p']:.4f}")
    
    return results


def analyze_sample(filepath, sample_name):
    print(f"\n{'='*70}")
    print(f"SAMPLE: {sample_name}")
    print(f"{'='*70}")
    
    df = load_data(filepath)
    n_msgs = len(df)
    n_convs = df['conversation_id'].nunique()
    print(f"Messages: {n_msgs}, Conversations: {n_convs}")
    
    all_results = []
    for measure_name, measure_col in MEASURES.items():
        result = analyze_measure(df, measure_name, measure_col)
        if result:
            result['sample'] = sample_name
            result['n_messages'] = n_msgs
            all_results.append(result)
    
    return pd.DataFrame(all_results)


# ============================================================================
# OUTPUT FORMATTING
# ============================================================================

def format_sample_results(sample_df, sample_name):
    """Format results for a single sample as text."""
    lines = []
    
    if len(sample_df) == 0:
        return lines
    
    lines.append(f"\n{'='*80}")
    lines.append(f"SAMPLE: {sample_name}")
    lines.append(f"{'='*80}")
    
    # Get sample stats
    n_conv = sample_df.iloc[0].get('n_conversations', 'N/A')
    n_exch = sample_df.iloc[0].get('n_exchanges', 'N/A')
    n_msg = sample_df.iloc[0].get('n_messages', 'N/A')
    lines.append(f"Messages: {n_msg}, Conversations: {n_conv}, Exchanges: {n_exch}")
    
    # IMMEDIATE Effects
    lines.append(f"\n## IMMEDIATE Response Effects")
    lines.append(f"   AI_t ~ User_t + AI_{{t-1}}  |  User_{{t+1}} ~ AI_t + User_t")
    lines.append("-" * 80)
    lines.append(f"{'Measure':<18} {'U→AI(imm) β':<14} {'p':<10} {'AI→U(imm) β':<14} {'p':<10}")
    lines.append("-" * 80)
    
    for _, row in sample_df.iterrows():
        u2a = row.get('ai_imm_user_beta', np.nan)
        u2a_p = row.get('ai_imm_user_p', np.nan)
        a2u = row.get('user_imm_ai_beta', np.nan)
        a2u_p = row.get('user_imm_ai_p', np.nan)
        
        u2a_sig = "*" if not np.isnan(u2a_p) and u2a_p < 0.05 else ""
        a2u_sig = "*" if not np.isnan(a2u_p) and a2u_p < 0.05 else ""
        
        u2a_str = f"{u2a:.4f}{u2a_sig}" if not np.isnan(u2a) else "N/A"
        a2u_str = f"{a2u:.4f}{a2u_sig}" if not np.isnan(a2u) else "N/A"
        u2a_p_str = f"{u2a_p:.4f}" if not np.isnan(u2a_p) else "N/A"
        a2u_p_str = f"{a2u_p:.4f}" if not np.isnan(a2u_p) else "N/A"
        
        lines.append(f"{row['measure']:<18} {u2a_str:<14} {u2a_p_str:<10} {a2u_str:<14} {a2u_p_str:<10}")
    
    # CROSS-LAG Effects
    lines.append(f"\n## CROSS-LAG Effects (Sustained Attunement)")
    lines.append(f"   AI_t ~ User_{{t-1}} + AI_{{t-1}}  |  User_t ~ AI_{{t-1}} + User_{{t-1}}")
    lines.append("-" * 80)
    lines.append(f"{'Measure':<18} {'U→AI(lag) β':<14} {'p':<10} {'AI→U(lag) β':<14} {'p':<10}")
    lines.append("-" * 80)
    
    for _, row in sample_df.iterrows():
        u2a = row.get('ai_lag_user_beta', np.nan)
        u2a_p = row.get('ai_lag_user_p', np.nan)
        a2u = row.get('user_lag_ai_beta', np.nan)
        a2u_p = row.get('user_lag_ai_p', np.nan)
        
        u2a_sig = "*" if not np.isnan(u2a_p) and u2a_p < 0.05 else ""
        a2u_sig = "*" if not np.isnan(a2u_p) and a2u_p < 0.05 else ""
        
        u2a_str = f"{u2a:.4f}{u2a_sig}" if not np.isnan(u2a) else "N/A"
        a2u_str = f"{a2u:.4f}{a2u_sig}" if not np.isnan(a2u) else "N/A"
        u2a_p_str = f"{u2a_p:.4f}" if not np.isnan(u2a_p) else "N/A"
        a2u_p_str = f"{a2u_p:.4f}" if not np.isnan(a2u_p) else "N/A"
        
        lines.append(f"{row['measure']:<18} {u2a_str:<14} {u2a_p_str:<10} {a2u_str:<14} {a2u_p_str:<10}")
    
    # COMPARISON: Immediate vs Cross-Lag
    lines.append(f"\n## COMPARISON: Immediate vs Cross-Lag")
    lines.append("-" * 90)
    lines.append(f"{'Measure':<18} {'U→AI imm':<10} {'U→AI lag':<10} {'Δ':<10} {'AI→U imm':<10} {'AI→U lag':<10} {'Δ':<10}")
    lines.append("-" * 90)
    
    for _, row in sample_df.iterrows():
        imm_u2a = row.get('ai_imm_user_beta', np.nan)
        lag_u2a = row.get('ai_lag_user_beta', np.nan)
        imm_a2u = row.get('user_imm_ai_beta', np.nan)
        lag_a2u = row.get('user_lag_ai_beta', np.nan)
        
        delta_u2a = imm_u2a - lag_u2a if not (np.isnan(imm_u2a) or np.isnan(lag_u2a)) else np.nan
        delta_a2u = imm_a2u - lag_a2u if not (np.isnan(imm_a2u) or np.isnan(lag_a2u)) else np.nan
        
        imm_u2a_str = f"{imm_u2a:.4f}" if not np.isnan(imm_u2a) else "N/A"
        lag_u2a_str = f"{lag_u2a:.4f}" if not np.isnan(lag_u2a) else "N/A"
        delta_u2a_str = f"{delta_u2a:+.4f}" if not np.isnan(delta_u2a) else "N/A"
        imm_a2u_str = f"{imm_a2u:.4f}" if not np.isnan(imm_a2u) else "N/A"
        lag_a2u_str = f"{lag_a2u:.4f}" if not np.isnan(lag_a2u) else "N/A"
        delta_a2u_str = f"{delta_a2u:+.4f}" if not np.isnan(delta_a2u) else "N/A"
        
        lines.append(f"{row['measure']:<18} {imm_u2a_str:<10} {lag_u2a_str:<10} {delta_u2a_str:<10} "
                    f"{imm_a2u_str:<10} {lag_a2u_str:<10} {delta_a2u_str:<10}")
    
    return lines


def write_comprehensive_report(results_df, output_file, samples_info):
    """Write report with all samples' immediate and cross-lag results."""
    
    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("EMOTIONAL ATTUNEMENT ANALYSIS: IMMEDIATE & CROSS-LAG EFFECTS\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("MODEL SPECIFICATION\n")
        f.write("-" * 80 + "\n")
        f.write("IMMEDIATE (same-turn alignment):\n")
        f.write("  AI equation:   AI_t ~ User_t + AI_{t-1}\n")
        f.write("  User equation: User_{t+1} ~ AI_t + User_t\n\n")
        f.write("CROSS-LAG (sustained attunement):\n")
        f.write("  AI equation:   AI_t ~ User_{t-1} + AI_{t-1}\n")
        f.write("  User equation: User_t ~ AI_{t-1} + User_{t-1}\n\n")
        f.write("All coefficients are STANDARDIZED (β*) and WITHIN-CONVERSATION CENTERED\n")
        f.write("* indicates p < 0.05\n\n")
        
        f.write("SAMPLES ANALYZED\n")
        f.write("-" * 80 + "\n")
        for filepath, name in samples_info:
            f.write(f"  {name}: {os.path.basename(filepath)}\n")
        f.write("\n")
        
        # Write results for each sample
        sample_names = results_df['sample'].unique()
        for sample in sample_names:
            sample_df = results_df[results_df['sample'] == sample]
            lines = format_sample_results(sample_df, sample)
            for line in lines:
                f.write(line + "\n")
        
        # Summary comparison table (RoBERTa only)
        f.write("\n" + "=" * 80 + "\n")
        f.write("SUMMARY COMPARISON (RoBERTa_Sum only)\n")
        f.write("=" * 80 + "\n\n")
        
        roberta_df = results_df[results_df['measure'] == 'RoBERTa_Sum']
        
        f.write(f"{'Sample':<20} {'U→AI imm':<12} {'U→AI lag':<12} {'AI→U imm':<12} {'AI→U lag':<12}\n")
        f.write("-" * 80 + "\n")
        
        for sample in sample_names:
            row = roberta_df[roberta_df['sample'] == sample]
            if len(row) == 0:
                continue
            row = row.iloc[0]
            
            imm_u2a = row.get('ai_imm_user_beta', np.nan)
            lag_u2a = row.get('ai_lag_user_beta', np.nan)
            imm_a2u = row.get('user_imm_ai_beta', np.nan)
            lag_a2u = row.get('user_lag_ai_beta', np.nan)
            
            imm_u2a_p = row.get('ai_imm_user_p', np.nan)
            lag_u2a_p = row.get('ai_lag_user_p', np.nan)
            imm_a2u_p = row.get('user_imm_ai_p', np.nan)
            lag_a2u_p = row.get('user_lag_ai_p', np.nan)
            
            def fmt(val, p):
                if np.isnan(val):
                    return "N/A"
                sig = "*" if not np.isnan(p) and p < 0.05 else ""
                return f"{val:.4f}{sig}"
            
            f.write(f"{sample:<20} {fmt(imm_u2a, imm_u2a_p):<12} {fmt(lag_u2a, lag_u2a_p):<12} "
                   f"{fmt(imm_a2u, imm_a2u_p):<12} {fmt(lag_a2u, lag_a2u_p):<12}\n")


# ============================================================================
# MAIN
# ============================================================================

def main(samples=None, batch_mode=False):
    """
    Run immediate response analysis on provided samples.
    
    Args:
        samples: List of (file_path, analysis_name) tuples, or None for defaults
        batch_mode: If True, use BATCH_SAMPLES for model comparison
    """
    print("=" * 80)
    print("EMOTIONAL ATTUNEMENT ANALYSIS: IMMEDIATE & CROSS-LAG EFFECTS")
    print("=" * 80)
    print(f"\nTimestamp: {TIMESTAMP}")
    print("\nModel Specification:")
    print("  IMMEDIATE: AI_t ~ User_t + AI_{t-1}  (AI responds to CURRENT user)")
    print("  CROSS-LAG: AI_t ~ User_{t-1} + AI_{t-1}  (AI attunes to PREVIOUS user)")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Determine which samples to use
    if batch_mode:
        samples = BATCH_SAMPLES
        output_name = "batch_model_comparison"
    elif samples is None:
        samples = DEFAULT_SAMPLES
        output_name = "default"
    else:
        output_name = "_".join([name for _, name in samples])
        if len(output_name) > 50:
            output_name = output_name[:50]
    
    # Analyze all samples
    all_results = []
    valid_samples = []
    
    for filepath, analysis_name in samples:
        if not os.path.exists(filepath):
            print(f"\nWARNING: File not found: {filepath}")
            continue
        result_df = analyze_sample(filepath, analysis_name)
        if len(result_df) > 0:
            all_results.append(result_df)
            valid_samples.append((filepath, analysis_name))
    
    if not all_results:
        print("No results generated!")
        return None
    
    results_df = pd.concat(all_results, ignore_index=True)
    
    # Save CSV
    csv_file = os.path.join(OUTPUT_DIR, f"results_{output_name}_{TIMESTAMP}.csv")
    results_df.to_csv(csv_file, index=False)
    
    # Save comprehensive TXT report
    txt_file = os.path.join(OUTPUT_DIR, f"report_{output_name}_{TIMESTAMP}.txt")
    write_comprehensive_report(results_df, txt_file, valid_samples)
    
    # Print summary to console
    print("\n" + "=" * 80)
    print("SUMMARY (RoBERTa_Sum)")
    print("=" * 80)
    
    roberta_df = results_df[results_df['measure'] == 'RoBERTa_Sum']
    print(f"\n{'Sample':<20} {'U→AI imm':<12} {'U→AI lag':<12} {'AI→U imm':<12} {'AI→U lag':<12}")
    print("-" * 80)
    
    for sample in results_df['sample'].unique():
        row = roberta_df[roberta_df['sample'] == sample]
        if len(row) == 0:
            continue
        row = row.iloc[0]
        
        def fmt(val, p):
            if np.isnan(val):
                return "N/A"
            sig = "*" if not np.isnan(p) and p < 0.05 else ""
            return f"{val:.4f}{sig}"
        
        print(f"{sample:<20} "
              f"{fmt(row.get('ai_imm_user_beta', np.nan), row.get('ai_imm_user_p', np.nan)):<12} "
              f"{fmt(row.get('ai_lag_user_beta', np.nan), row.get('ai_lag_user_p', np.nan)):<12} "
              f"{fmt(row.get('user_imm_ai_beta', np.nan), row.get('user_imm_ai_p', np.nan)):<12} "
              f"{fmt(row.get('user_lag_ai_beta', np.nan), row.get('user_lag_ai_p', np.nan)):<12}")
    
    print(f"\n{'='*80}")
    print("FILES SAVED")
    print(f"{'='*80}")
    print(f"  CSV: {csv_file}")
    print(f"  TXT: {txt_file}")
    
    return results_df


def parse_args():
    parser = argparse.ArgumentParser(description='Immediate & Cross-Lag Response Analysis')
    parser.add_argument('file_path', nargs='?', help='Path to input CSV file')
    parser.add_argument('analysis_name', nargs='?', help='Name for this analysis')
    parser.add_argument('--batch', action='store_true', help='Run batch mode (all model comparison files)')
    parser.add_argument('--samples', nargs='*', help='Multiple samples as file:name pairs')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    if args.batch:
        # Batch mode: run all predefined model comparison files
        results = main(batch_mode=True)
    elif args.samples:
        # Parse multiple samples from --samples file1:name1 file2:name2
        samples = []
        for s in args.samples:
            if ':' in s:
                fpath, name = s.split(':', 1)
                samples.append((fpath, name))
        results = main(samples)
    elif args.file_path and args.analysis_name:
        # Single sample from positional arguments
        results = main([(args.file_path, args.analysis_name)])
    else:
        # Default samples
        results = main()
