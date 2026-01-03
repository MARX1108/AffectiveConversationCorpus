"""
Conversation-Level Emotional Association Analysis
"""

import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from datetime import datetime
import os
import warnings

warnings.filterwarnings('ignore')

FULL_SAMPLE = ""
ANNOTATED_SAMPLE = ""
OUTPUT_DIR = ""

MEASURES = {
    'VADER': 'vader_compound',
    'NRC_Valence': 'nrc_valence',
    'NRC_Arousal': 'nrc_arousal',
    'NRC_Dominance': 'nrc_dominance',
    'RoBERTa': 'roberta_valence'
}


def aggregate_by_conversation(df):
    agg_cols = {col: 'mean' for col in MEASURES.values() if col in df.columns}
    agg_cols['message'] = 'count'
    
    agg = df.groupby(['conversation_id', 'user_or_ai']).agg(agg_cols).reset_index()
    agg = agg.rename(columns={'message': 'n_messages'})
    
    user_df = agg[agg['user_or_ai'] == 'user'].copy()
    ai_df = agg[agg['user_or_ai'] == 'ai'].copy()
    
    user_df = user_df.drop(columns=['user_or_ai'])
    ai_df = ai_df.drop(columns=['user_or_ai'])
    
    user_df.columns = ['conversation_id'] + [f'user_{c}' if c != 'conversation_id' else c for c in user_df.columns[1:]]
    ai_df.columns = ['conversation_id'] + [f'ai_{c}' if c != 'conversation_id' else c for c in ai_df.columns[1:]]
    
    merged = user_df.merge(ai_df, on='conversation_id', how='inner')
    return merged


def run_correlation_tests(conv_df, measure_name, measure_col):
    user_col = f'user_{measure_col}'
    ai_col = f'ai_{measure_col}'
    
    if user_col not in conv_df.columns or ai_col not in conv_df.columns:
        return None
    
    clean = conv_df[[user_col, ai_col]].dropna()
    n = len(clean)
    
    if n < 10:
        return None
    
    r_pearson, p_pearson = stats.pearsonr(clean[user_col], clean[ai_col])
    r_spearman, p_spearman = stats.spearmanr(clean[user_col], clean[ai_col])
    
    X = sm.add_constant(clean[user_col])
    model = sm.OLS(clean[ai_col], X).fit()
    
    beta = model.params[user_col]
    se = model.bse[user_col]
    t = model.tvalues[user_col]
    p_reg = model.pvalues[user_col]
    r2 = model.rsquared
    
    X_rev = sm.add_constant(clean[ai_col])
    model_rev = sm.OLS(clean[user_col], X_rev).fit()
    beta_rev = model_rev.params[ai_col]
    
    return {
        'measure': measure_name,
        'n_conversations': n,
        'user_mean': clean[user_col].mean(),
        'user_sd': clean[user_col].std(),
        'ai_mean': clean[ai_col].mean(),
        'ai_sd': clean[ai_col].std(),
        'pearson_r': r_pearson,
        'pearson_p': p_pearson,
        'spearman_r': r_spearman,
        'spearman_p': p_spearman,
        'beta_user_to_ai': beta,
        'se_user_to_ai': se,
        't_user_to_ai': t,
        'p_user_to_ai': p_reg,
        'beta_ai_to_user': beta_rev,
        'r_squared': r2
    }


def run_paired_tests(conv_df, measure_name, measure_col):
    user_col = f'user_{measure_col}'
    ai_col = f'ai_{measure_col}'
    
    if user_col not in conv_df.columns or ai_col not in conv_df.columns:
        return None
    
    clean = conv_df[[user_col, ai_col]].dropna()
    n = len(clean)
    
    if n < 10:
        return None
    
    t_stat, p_paired = stats.ttest_rel(clean[user_col], clean[ai_col])
    diff_mean = (clean[user_col] - clean[ai_col]).mean()
    diff_sd = (clean[user_col] - clean[ai_col]).std()
    
    return {
        'measure': measure_name,
        'n_conversations': n,
        'user_mean': clean[user_col].mean(),
        'ai_mean': clean[ai_col].mean(),
        'mean_difference': diff_mean,
        'sd_difference': diff_sd,
        't_statistic': t_stat,
        'p_value': p_paired,
        'cohens_d': diff_mean / diff_sd if diff_sd > 0 else 0
    }


def analyze_sample(df, sample_name):
    print(f"\n{'='*60}")
    print(f"SAMPLE: {sample_name}")
    print(f"{'='*60}")
    
    conv_df = aggregate_by_conversation(df)
    print(f"Conversations: {len(conv_df)}")
    
    correlation_results = []
    paired_results = []
    
    print(f"\n--- Correlation Analysis ---")
    print(f"{'Measure':<15} {'N':>5} {'Pearson r':>10} {'p':>8} {'Beta(U→A)':>10} {'R²':>8}")
    print("-" * 60)
    
    for measure_name, measure_col in MEASURES.items():
        result = run_correlation_tests(conv_df, measure_name, measure_col)
        if result:
            correlation_results.append(result)
            sig = "*" if result['pearson_p'] < 0.05 else ""
            print(f"{measure_name:<15} {result['n_conversations']:>5} {result['pearson_r']:>10.4f}{sig} "
                  f"{result['pearson_p']:>8.4f} {result['beta_user_to_ai']:>10.4f} {result['r_squared']:>8.4f}")
    
    print(f"\n--- Paired Comparison (User vs AI) ---")
    print(f"{'Measure':<15} {'User M':>8} {'AI M':>8} {'Diff':>8} {'t':>8} {'p':>8} {'d':>8}")
    print("-" * 70)
    
    for measure_name, measure_col in MEASURES.items():
        result = run_paired_tests(conv_df, measure_name, measure_col)
        if result:
            paired_results.append(result)
            sig = "*" if result['p_value'] < 0.05 else ""
            print(f"{measure_name:<15} {result['user_mean']:>8.4f} {result['ai_mean']:>8.4f} "
                  f"{result['mean_difference']:>8.4f} {result['t_statistic']:>8.2f} "
                  f"{result['p_value']:>8.4f}{sig} {result['cohens_d']:>8.4f}")
    
    return conv_df, correlation_results, paired_results


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("="*60)
    print("CONVERSATION-LEVEL EMOTIONAL ASSOCIATION ANALYSIS")
    print("="*60)
    
    df_full = pd.read_csv(FULL_SAMPLE)
    df_annotated = pd.read_csv(ANNOTATED_SAMPLE)
    
    if 'roberta_valence' not in df_annotated.columns:
        print("\nNote: RoBERTa scores not in annotated sample, will be skipped")
    
    conv_full, corr_full, paired_full = analyze_sample(df_full, f"Full Sample (N={df_full['conversation_id'].nunique()})")
    conv_ann, corr_ann, paired_ann = analyze_sample(df_annotated, f"Annotated Sample (N={df_annotated['conversation_id'].nunique()})")
    
    conv_full.to_csv(f"{OUTPUT_DIR}/conversation_aggregates_{timestamp}.csv", index=False)
    
    all_corr = []
    for r in corr_full:
        r['sample'] = 'full'
        all_corr.append(r)
    for r in corr_ann:
        r['sample'] = 'annotated'
        all_corr.append(r)
    
    pd.DataFrame(all_corr).to_csv(f"{OUTPUT_DIR}/correlation_results_{timestamp}.csv", index=False)
    
    all_paired = []
    for r in paired_full:
        r['sample'] = 'full'
        all_paired.append(r)
    for r in paired_ann:
        r['sample'] = 'annotated'
        all_paired.append(r)
    
    pd.DataFrame(all_paired).to_csv(f"{OUTPUT_DIR}/paired_comparison_results_{timestamp}.csv", index=False)
    
    with open(f"{OUTPUT_DIR}/summary_{timestamp}.txt", 'w') as f:
        f.write("CONVERSATION-LEVEL EMOTIONAL ASSOCIATION ANALYSIS\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("CORRELATION RESULTS\n")
        f.write("-" * 40 + "\n\n")
        
        f.write("Full Sample:\n")
        for r in corr_full:
            sig = "*" if r['pearson_p'] < 0.05 else ""
            f.write(f"  {r['measure']}: r={r['pearson_r']:.4f}{sig}, p={r['pearson_p']:.4f}, "
                   f"β={r['beta_user_to_ai']:.4f}, R²={r['r_squared']:.4f}\n")
        
        f.write("\nAnnotated Sample:\n")
        for r in corr_ann:
            sig = "*" if r['pearson_p'] < 0.05 else ""
            f.write(f"  {r['measure']}: r={r['pearson_r']:.4f}{sig}, p={r['pearson_p']:.4f}, "
                   f"β={r['beta_user_to_ai']:.4f}, R²={r['r_squared']:.4f}\n")
        
        f.write("\n\nPAIRED COMPARISON RESULTS\n")
        f.write("-" * 40 + "\n\n")
        
        f.write("Full Sample:\n")
        for r in paired_full:
            sig = "*" if r['p_value'] < 0.05 else ""
            f.write(f"  {r['measure']}: User M={r['user_mean']:.4f}, AI M={r['ai_mean']:.4f}, "
                   f"diff={r['mean_difference']:.4f}, t={r['t_statistic']:.2f}, p={r['p_value']:.4f}{sig}\n")
        
        f.write("\nAnnotated Sample:\n")
        for r in paired_ann:
            sig = "*" if r['p_value'] < 0.05 else ""
            f.write(f"  {r['measure']}: User M={r['user_mean']:.4f}, AI M={r['ai_mean']:.4f}, "
                   f"diff={r['mean_difference']:.4f}, t={r['t_statistic']:.2f}, p={r['p_value']:.4f}{sig}\n")
    
    print(f"\n{'='*60}")
    print("FILES SAVED")
    print(f"{'='*60}")
    print(f"  {OUTPUT_DIR}/conversation_aggregates_{timestamp}.csv")
    print(f"  {OUTPUT_DIR}/correlation_results_{timestamp}.csv")
    print(f"  {OUTPUT_DIR}/paired_comparison_results_{timestamp}.csv")
    print(f"  {OUTPUT_DIR}/summary_{timestamp}.txt")


if __name__ == "__main__":
    main()

