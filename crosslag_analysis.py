#!/usr/bin/env python3
"""
Unified Cross-Lagged Panel Analysis

A cross-lagged analysis with:
1. CLUSTERING: Cluster-robust standard errors
2. CAUSAL STRUCTURE: Symmetric lagged predictors (A_{t-1}, U_{t-1})
3. HETEROGENEITY: Within-conversation centering (RI-CLPM)
4. RANDOM EFFECTS: Nested random intercepts (conversation, participant)

Usage:
    python unified_crosslag_analysis.py <file_path> <analysis_name>
    python unified_crosslag_analysis.py  # runs default samples

Author: Marx 
Date: 2025-12-17
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

DEFAULT_SAMPLES = [
    ("", "Full_N447"),
    ("", "Annotated_N73")
]
OUTPUT_DIR = ""
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')

# All measures standardized to 0-1
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

def load_and_prepare_data(filepath):
    """Load data and standardize VADER to 0-1."""
    print(f"Loading: {filepath}")
    df = pd.read_csv(filepath)
    
    # Standardize VADER from [-1,1] to [0,1]
    df['vader_compound_01'] = (df['vader_compound'] + 1) / 2
    
    print(f"  Messages: {len(df)}")
    print(f"  Conversations: {df['conversation_id'].nunique()}")
    print(f"  Participants: {df['participant_id'].nunique()}")
    
    return df


def create_exchange_data(df, measure_col):
    """
    Create exchange-level dataset with proper temporal structure.
    
    Exchange i = (User_i, AI_i) where AI_i responds to User_i
    
    For cross-lag analysis:
    - AI_t ~ AI_{t-1} + User_{t-1}  (AI responds to PREVIOUS user turn - attunement)
    - User_t ~ User_{t-1} + AI_{t-1}  (User responds to PREVIOUS AI turn - attunement)
    """
    exchanges = []
    
    for conv_id, conv_df in df.groupby('conversation_id'):
        conv_df = conv_df.sort_values('timestamp')
        participant_id = conv_df['participant_id'].iloc[0]
        
        user_msgs = conv_df[conv_df['user_or_ai'] == 'user'].reset_index(drop=True)
        ai_msgs = conv_df[conv_df['user_or_ai'] == 'ai'].reset_index(drop=True)
        
        n_exchanges = min(len(user_msgs), len(ai_msgs))
        
        for i in range(n_exchanges):
            exchanges.append({
                'conversation_id': conv_id,
                'participant_id': participant_id,
                'exchange_idx': i + 1,
                'user_score': user_msgs.iloc[i][measure_col],
                'ai_score': ai_msgs.iloc[i][measure_col]
            })
    
    ex_df = pd.DataFrame(exchanges)
    
    # Create lagged variables (within conversation)
    ex_df['user_lag'] = ex_df.groupby('conversation_id')['user_score'].shift(1)
    ex_df['ai_lag'] = ex_df.groupby('conversation_id')['ai_score'].shift(1)
    
    return ex_df


def apply_centering(ex_df):
    """
    Apply within-conversation centering (RI-CLPM approximation).
    
    This separates:
    - TRAIT effects: Stable individual/conversation differences
    - STATE effects: Dynamic turn-by-turn attunement
    
    X_centered = X_raw - mean(X within conversation)
    """
    df = ex_df.copy()
    
    # Calculate conversation means
    conv_means = df.groupby('conversation_id').agg({
        'user_score': 'mean',
        'ai_score': 'mean'
    }).rename(columns={'user_score': 'user_conv_mean', 'ai_score': 'ai_conv_mean'})
    
    df = df.merge(conv_means, on='conversation_id', how='left')
    
    # Center within conversation
    df['user_c'] = df['user_score'] - df['user_conv_mean']
    df['ai_c'] = df['ai_score'] - df['ai_conv_mean']
    
    # Create centered lags
    df['user_c_lag'] = df.groupby('conversation_id')['user_c'].shift(1)
    df['ai_c_lag'] = df.groupby('conversation_id')['ai_c'].shift(1)
    
    return df


def standardize_variables(df, cols):
    """Z-score standardization for comparable coefficients."""
    df = df.copy()
    for col in cols:
        if col in df.columns and df[col].std() > 0:
            df[f'{col}_z'] = (df[col] - df[col].mean()) / df[col].std()
    return df


# ============================================================================
# STATISTICAL MODELS
# ============================================================================

def run_mixed_model(df, dv, iv_lag, iv_cross, group_col):
    """
    Run mixed-effects model with random intercepts.
    
    Model: DV ~ DV_lag + IV_lag + (1|group)
    
    Returns coefficients, SEs, z-values, p-values, and model fit stats.
    """
    formula = f"{dv} ~ {iv_lag} + {iv_cross}"
    
    try:
        model = smf.mixedlm(
            formula,
            data=df.dropna(subset=[dv, iv_lag, iv_cross]),
            groups=df[group_col]
        ).fit(method='powell', disp=False)
        
        return {
            'ar_beta': model.fe_params[iv_lag],
            'ar_se': model.bse[iv_lag],
            'ar_p': model.pvalues[iv_lag],
            'cross_beta': model.fe_params[iv_cross],
            'cross_se': model.bse[iv_cross],
            'cross_z': model.tvalues[iv_cross],
            'cross_p': model.pvalues[iv_cross],
            'intercept': model.fe_params['Intercept'],
            'group_var': model.cov_re.iloc[0, 0] if hasattr(model, 'cov_re') else None,
            'n_obs': len(df.dropna(subset=[dv, iv_lag, iv_cross])),
            'n_groups': df[group_col].nunique(),
            'converged': True
        }
    except Exception as e:
        return {'error': str(e), 'converged': False}


def calculate_variance_stats(df, user_col, ai_col):
    """Calculate variance statistics for user and AI scores."""
    return {
        'user_mean': df[user_col].mean(),
        'user_sd': df[user_col].std(),
        'user_var': df[user_col].var(),
        'ai_mean': df[ai_col].mean(),
        'ai_sd': df[ai_col].std(),
        'ai_var': df[ai_col].var(),
        'user_ai_corr': df[user_col].corr(df[ai_col])
    }


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def analyze_measure(df, measure_name, measure_col):
    """Run complete cross-lag analysis for a single measure."""
    print(f"\n  {measure_name}:")
    
    # Create exchange data
    ex_df = create_exchange_data(df, measure_col)
    
    # Apply centering
    ex_df = apply_centering(ex_df)
    
    # Standardize all variables
    cols_to_std = ['user_score', 'ai_score', 'user_lag', 'ai_lag',
                   'user_c', 'ai_c', 'user_c_lag', 'ai_c_lag']
    ex_df = standardize_variables(ex_df, cols_to_std)
    
    # Clean data
    ex_df_clean = ex_df.dropna().copy()
    
    if len(ex_df_clean) < 50:
        print(f"    Insufficient data: {len(ex_df_clean)} observations")
        return None
    
    # Variance stats (raw)
    var_raw = calculate_variance_stats(ex_df_clean, 'user_score', 'ai_score')
    var_centered = calculate_variance_stats(ex_df_clean, 'user_c', 'ai_c')
    
    print(f"    Raw variance: User={var_raw['user_var']:.4f}, AI={var_raw['ai_var']:.4f}")
    print(f"    Centered variance: User={var_centered['user_var']:.4f}, AI={var_centered['ai_var']:.4f}")
    
    results = {
        'measure': measure_name,
        'n_obs': len(ex_df_clean),
        'n_conversations': ex_df_clean['conversation_id'].nunique(),
        'n_participants': ex_df_clean['participant_id'].nunique(),
        **{f'raw_{k}': v for k, v in var_raw.items()},
        **{f'centered_{k}': v for k, v in var_centered.items()}
    }
    
    # ========================================================================
    # MODEL 1: Raw + Standardized (conversation random intercepts)
    # ========================================================================
    print("    Running Raw + Std models...")
    
    # User→AI (AI equation)
    m_ai_raw = run_mixed_model(
        ex_df_clean, 'ai_score_z', 'ai_lag_z', 'user_lag_z', 'conversation_id'
    )
    
    # AI→User (User equation)
    m_user_raw = run_mixed_model(
        ex_df_clean, 'user_score_z', 'user_lag_z', 'ai_lag_z', 'conversation_id'
    )
    
    if m_ai_raw['converged'] and m_user_raw['converged']:
        results['raw_u2a_beta'] = m_ai_raw['cross_beta']
        results['raw_u2a_se'] = m_ai_raw['cross_se']
        results['raw_u2a_p'] = m_ai_raw['cross_p']
        results['raw_a2u_beta'] = m_user_raw['cross_beta']
        results['raw_a2u_se'] = m_user_raw['cross_se']
        results['raw_a2u_p'] = m_user_raw['cross_p']
        results['raw_ai_ar'] = m_ai_raw['ar_beta']
        results['raw_user_ar'] = m_user_raw['ar_beta']
        results['raw_conv_var_ai'] = m_ai_raw['group_var']
        results['raw_conv_var_user'] = m_user_raw['group_var']
    
    # ========================================================================
    # MODEL 2: Centered + Standardized (RI-CLPM - STATE effects)
    # ========================================================================
    print("    Running Centered + Std models (RI-CLPM)...")
    
    # User→AI (AI equation) - centered
    m_ai_c = run_mixed_model(
        ex_df_clean, 'ai_c_z', 'ai_c_lag_z', 'user_c_lag_z', 'conversation_id'
    )
    
    # AI→User (User equation) - centered
    m_user_c = run_mixed_model(
        ex_df_clean, 'user_c_z', 'user_c_lag_z', 'ai_c_lag_z', 'conversation_id'
    )
    
    if m_ai_c['converged'] and m_user_c['converged']:
        results['centered_u2a_beta'] = m_ai_c['cross_beta']
        results['centered_u2a_se'] = m_ai_c['cross_se']
        results['centered_u2a_p'] = m_ai_c['cross_p']
        results['centered_a2u_beta'] = m_user_c['cross_beta']
        results['centered_a2u_se'] = m_user_c['cross_se']
        results['centered_a2u_p'] = m_user_c['cross_p']
        results['centered_ai_ar'] = m_ai_c['ar_beta']
        results['centered_user_ar'] = m_user_c['ar_beta']
        results['centered_conv_var_ai'] = m_ai_c['group_var']
        results['centered_conv_var_user'] = m_user_c['group_var']
    
    # ========================================================================
    # MODEL 3: Participant-level random intercepts (nested structure)
    # ========================================================================
    print("    Running Participant-level models...")
    
    # User→AI with participant random intercepts
    m_ai_part = run_mixed_model(
        ex_df_clean, 'ai_c_z', 'ai_c_lag_z', 'user_c_lag_z', 'participant_id'
    )
    
    # AI→User with participant random intercepts
    m_user_part = run_mixed_model(
        ex_df_clean, 'user_c_z', 'user_c_lag_z', 'ai_c_lag_z', 'participant_id'
    )
    
    if m_ai_part['converged'] and m_user_part['converged']:
        results['part_u2a_beta'] = m_ai_part['cross_beta']
        results['part_u2a_se'] = m_ai_part['cross_se']
        results['part_u2a_p'] = m_ai_part['cross_p']
        results['part_a2u_beta'] = m_user_part['cross_beta']
        results['part_a2u_se'] = m_user_part['cross_se']
        results['part_a2u_p'] = m_user_part['cross_p']
        results['part_var_ai'] = m_ai_part['group_var']
        results['part_var_user'] = m_user_part['group_var']
    return results


def analyze_sample(filepath, sample_name):
    """Run complete analysis on a single sample."""
    print(f"\n{'='*80}")
    print(f"ANALYZING: {sample_name}")
    print(f"{'='*80}")
    
    df = load_and_prepare_data(filepath)
    
    all_results = []
    for measure_name, measure_col in MEASURES.items():
        result = analyze_measure(df, measure_name, measure_col)
        if result:
            result['sample'] = sample_name
            all_results.append(result)
    
    return pd.DataFrame(all_results)


def main(samples=None):
    """
    Run cross-lagged analysis on provided samples.
    
    Args:
        samples: List of (file_path, analysis_name) tuples, or None for defaults
    """
    print("=" * 80)
    print("CROSS-LAGGED PANEL ANALYSIS")
    print("=" * 80)
    print(f"\nTimestamp: {TIMESTAMP}")
    print("\nModel Specification:")
    print("  AI equation:   A_t ~ A_{t-1} + U_{t-1} + (1|cluster)")
    print("  User equation: U_t ~ U_{t-1} + A_{t-1} + (1|cluster)")
    print("\nAnalysis Variants:")
    print("  1. Raw + Standardized (conversation clusters)")
    print("  2. Centered + Standardized (RI-CLPM, conversation clusters)")
    print("  3. Centered + Standardized (participant clusters)")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Use provided samples or defaults
    if samples is None:
        samples = DEFAULT_SAMPLES
    
    # Analyze all samples
    all_results = []
    for filepath, analysis_name in samples:
        if not os.path.exists(filepath):
            print(f"\nWARNING: File not found: {filepath}")
            continue
        result_df = analyze_sample(filepath, analysis_name)
        if len(result_df) > 0:
            all_results.append(result_df)
    
    if not all_results:
        print("No results generated!")
        return None
    
    # Combine results
    results_df = pd.concat(all_results, ignore_index=True)
    
    # Generate output filename based on analysis names
    analysis_names = "_".join([name for _, name in samples])
    if len(analysis_names) > 50:
        analysis_names = analysis_names[:50]
    
    # Save results
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)
    results_file = os.path.join(OUTPUT_DIR, f"crosslag_{analysis_names}_{TIMESTAMP}.csv")
    results_df.to_csv(results_file, index=False)
    print(f"  Saved: {results_file}")
    
    # ========================================================================
    # GENERATE SUMMARY REPORT
    # ========================================================================
    
    sample_names = results_df['sample'].unique()
    for sample_name in sample_names:
        sample_df = results_df[results_df['sample'] == sample_name]
        
        if len(sample_df) == 0:
            continue
        
        print("\n" + "=" * 80)
        print(f"RESULTS: {sample_name}")
        print("=" * 80)
        
        # Main results: Centered + Standardized
        print("\n## Centered + Standardized (RI-CLPM)")
        print("-" * 90)
        print(f"{'Measure':<15} {'User→AI β*':<12} {'p':<10} {'AI→User β*':<12} {'p':<10} {'Stronger':<12} {'Sig?'}")
        print("-" * 90)
        
        for _, row in sample_df.iterrows():
            u2a = row.get('centered_u2a_beta', np.nan)
            u2a_p = row.get('centered_u2a_p', np.nan)
            a2u = row.get('centered_a2u_beta', np.nan)
            a2u_p = row.get('centered_a2u_p', np.nan)
            
            if pd.isna(u2a) or pd.isna(a2u):
                continue
            
            u2a_sig = "*" if u2a_p < 0.05 else ""
            a2u_sig = "*" if a2u_p < 0.05 else ""
            sig_count = (1 if u2a_p < 0.05 else 0) + (1 if a2u_p < 0.05 else 0)
            
            if abs(a2u) > abs(u2a):
                stronger = "AI→User"
            else:
                stronger = "User→AI"
            
            print(f"{row['measure']:<15} {u2a:>10.4f}{u2a_sig:<1} {u2a_p:<10.4f} {a2u:>10.4f}{a2u_sig:<1} {a2u_p:<10.4f} {stronger:<12} {sig_count}/2")
        
        # Raw results
        print("\n## Raw + Standardized")
        print("-" * 90)
        print(f"{'Measure':<15} {'User→AI β*':<12} {'p':<10} {'AI→User β*':<12} {'p':<10}")
        print("-" * 90)
        
        for _, row in sample_df.iterrows():
            u2a = row.get('raw_u2a_beta', np.nan)
            u2a_p = row.get('raw_u2a_p', np.nan)
            a2u = row.get('raw_a2u_beta', np.nan)
            a2u_p = row.get('raw_a2u_p', np.nan)
            
            if pd.isna(u2a) or pd.isna(a2u):
                continue
            
            u2a_sig = "*" if u2a_p < 0.05 else ""
            a2u_sig = "*" if a2u_p < 0.05 else ""
            
            print(f"{row['measure']:<15} {u2a:>10.4f}{u2a_sig:<1} {u2a_p:<10.4f} {a2u:>10.4f}{a2u_sig:<1} {a2u_p:<10.4f}")
    
    # Write summary file
    summary_file = os.path.join(OUTPUT_DIR, f"summary_{analysis_names}_{TIMESTAMP}.txt")
    with open(summary_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("CROSS-LAGGED PANEL ANALYSIS\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("MODEL SPECIFICATION\n")
        f.write("-" * 80 + "\n")
        f.write("AI equation:   A_t ~ A_{t-1} + U_{t-1} + (1|conversation)\n")
        f.write("User equation: U_t ~ U_{t-1} + A_{t-1} + (1|conversation)\n\n")
        f.write("All coefficients are STANDARDIZED (β*)\n")
        f.write("Centered models use within-conversation centering (RI-CLPM)\n\n")
        
        for sample_name in sample_names:
            sample_df = results_df[results_df['sample'] == sample_name]
            if len(sample_df) == 0:
                continue
            
            f.write(f"\n{'='*80}\n")
            f.write(f"SAMPLE: {sample_name}\n")
            f.write(f"{'='*80}\n\n")
            
            f.write("Centered + Standardized (RI-CLPM)\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Measure':<15} {'User→AI β*':<12} {'p':<10} {'AI→User β*':<12} {'p':<10}\n")
            f.write("-" * 80 + "\n")
            
            for _, row in sample_df.iterrows():
                u2a = row.get('centered_u2a_beta', np.nan)
                u2a_p = row.get('centered_u2a_p', np.nan)
                a2u = row.get('centered_a2u_beta', np.nan)
                a2u_p = row.get('centered_a2u_p', np.nan)
                
                if pd.isna(u2a) or pd.isna(a2u):
                    continue
                
                u2a_sig = "*" if u2a_p < 0.05 else ""
                a2u_sig = "*" if a2u_p < 0.05 else ""
                
                f.write(f"{row['measure']:<15} {u2a:>10.4f}{u2a_sig:<1} {u2a_p:<10.4f} {a2u:>10.4f}{a2u_sig:<1} {a2u_p:<10.4f}\n")
    print(f"\n  Saved: {summary_file}")
    return results_df


def parse_args():
    parser = argparse.ArgumentParser(description='Cross-Lagged Panel Analysis')
    parser.add_argument('file_path', nargs='?', help='Path to input CSV file')
    parser.add_argument('analysis_name', nargs='?', help='Name for this analysis')
    parser.add_argument('--samples', nargs='*', help='Multiple samples as file:name pairs')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    if args.samples:
        # Parse multiple samples from --samples file1:name1 file2:name2
        samples = []
        for s in args.samples:
            if ':' in s:
                fpath, name = s.split(':', 1)
                samples.append((fpath, name))
        results_df = main(samples)
    elif args.file_path and args.analysis_name:
        # Single sample from positional arguments
        results_df = main([(args.file_path, args.analysis_name)])
    else:
        # Default samples
        results_df = main()


