#!/usr/bin/env python3
"""
Monte Carlo Power Analysis for Cross-Lagged Panel Models

This script performs power analysis for the emotional attunement cross-lagged
analysis using Monte Carlo simulation. It estimates:
1. Statistical power for observed effect sizes
2. Minimum detectable effect size (MDES) for 80% power
3. Required sample size for target power

Based on the affective sample parameters:
- N_conversations = 73
- Mean exchanges per conversation = 9.16
- Model: Mixed-effects with random intercepts (RI-CLPM approximation)

Date: 2025-12
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
import warnings
from datetime import datetime
import os
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

warnings.filterwarnings('ignore')

# =============================================================================
# SIMULATION PARAMETERS
# =============================================================================

# Observed parameters from affective sample
N_CONVERSATIONS = 73
MEAN_EXCHANGES = 9.16
SD_EXCHANGES = 4.0  # Estimated SD of exchanges per conversation

# Observed variance components (from RoBERTa analysis)
WITHIN_VAR = 0.08  # Within-conversation variance (state)
BETWEEN_VAR = 0.05  # Between-conversation variance (trait/random intercept)
RESIDUAL_VAR = 0.87  # Residual variance

# Autoregressive coefficients (observed)
AR_COEF = 0.15  # Autoregressive stability

# Effect sizes to test (standardized β*)
EFFECT_SIZES = [0.00, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30]

# Simulation settings
N_SIMULATIONS = 1000
ALPHA = 0.05
TARGET_POWER = 0.80

OUTPUT_DIR = "/Users/marxw/Mercury-Fyorin/acl_121125/results/power_analysis"
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')


# =============================================================================
# DATA GENERATION
# =============================================================================

def generate_crosslag_data(n_conv, mean_exch, sd_exch, cross_lag_effect, 
                           ar_coef=0.15, between_var=0.05, residual_var=0.87,
                           seed=None):
    """
    Generate simulated cross-lagged panel data.
    
    Model:
        Y_{c,t} = β0 + β1*Y_{c,t-1} + β2*X_{c,t-1} + u_c + ε_{c,t}
    
    where:
        - Y = AI sentiment (outcome)
        - X = User sentiment (predictor)
        - u_c = conversation random intercept
        - ε = residual
        - β2 = cross_lag_effect (the effect we're testing)
    
    Parameters:
    -----------
    n_conv : int
        Number of conversations
    mean_exch : float
        Mean number of exchanges per conversation
    sd_exch : float
        SD of exchanges per conversation
    cross_lag_effect : float
        True cross-lagged effect size (standardized)
    ar_coef : float
        Autoregressive coefficient
    between_var : float
        Between-conversation variance (random intercept)
    residual_var : float
        Residual variance
    seed : int
        Random seed
    
    Returns:
    --------
    pd.DataFrame with simulated data
    """
    if seed is not None:
        np.random.seed(seed)
    
    data = []
    
    for conv_id in range(n_conv):
        # Number of exchanges for this conversation (min 3)
        n_exch = max(3, int(np.random.normal(mean_exch, sd_exch)))
        
        # Conversation-level random intercepts
        u_user = np.random.normal(0, np.sqrt(between_var))
        u_ai = np.random.normal(0, np.sqrt(between_var))
        
        # Initialize
        user_scores = np.zeros(n_exch)
        ai_scores = np.zeros(n_exch)
        
        # First exchange (random)
        user_scores[0] = u_user + np.random.normal(0, np.sqrt(residual_var))
        ai_scores[0] = u_ai + np.random.normal(0, np.sqrt(residual_var))
        
        # Generate subsequent exchanges with cross-lagged structure
        for t in range(1, n_exch):
            # User score: AR + random intercept + residual
            user_scores[t] = (ar_coef * user_scores[t-1] + 
                              u_user + 
                              np.random.normal(0, np.sqrt(residual_var)))
            
            # AI score: AR + cross-lag from user + random intercept + residual
            ai_scores[t] = (ar_coef * ai_scores[t-1] + 
                            cross_lag_effect * user_scores[t-1] +  # Cross-lag effect
                            u_ai + 
                            np.random.normal(0, np.sqrt(residual_var)))
        
        for t in range(n_exch):
            data.append({
                'conversation_id': conv_id,
                'exchange_idx': t + 1,
                'user_score': user_scores[t],
                'ai_score': ai_scores[t]
            })
    
    df = pd.DataFrame(data)
    
    # Create lagged variables
    df['user_lag'] = df.groupby('conversation_id')['user_score'].shift(1)
    df['ai_lag'] = df.groupby('conversation_id')['ai_score'].shift(1)
    
    # Within-conversation centering (RI-CLPM)
    conv_means = df.groupby('conversation_id').agg({
        'user_score': 'mean',
        'ai_score': 'mean'
    }).rename(columns={'user_score': 'user_mean', 'ai_score': 'ai_mean'})
    
    df = df.merge(conv_means, on='conversation_id', how='left')
    df['user_c'] = df['user_score'] - df['user_mean']
    df['ai_c'] = df['ai_score'] - df['ai_mean']
    df['user_c_lag'] = df.groupby('conversation_id')['user_c'].shift(1)
    df['ai_c_lag'] = df.groupby('conversation_id')['ai_c'].shift(1)
    
    # Standardize
    for col in ['user_c', 'ai_c', 'user_c_lag', 'ai_c_lag']:
        if col in df.columns and df[col].std() > 0:
            df[f'{col}_z'] = (df[col] - df[col].mean()) / df[col].std()
    
    return df


# =============================================================================
# MODEL FITTING
# =============================================================================

def fit_crosslag_model(df):
    """
    Fit cross-lagged mixed-effects model.
    
    Model: AI_t ~ AI_{t-1} + User_{t-1} + (1|conversation)
    
    Returns:
    --------
    dict with coefficient estimates and p-values
    """
    try:
        clean_df = df.dropna(subset=['ai_c_z', 'ai_c_lag_z', 'user_c_lag_z']).copy()
        
        if len(clean_df) < 30:
            return {'converged': False}
        
        model = smf.mixedlm(
            "ai_c_z ~ ai_c_lag_z + user_c_lag_z",
            data=clean_df,
            groups=clean_df['conversation_id']
        ).fit(method='powell', disp=False)
        
        return {
            'converged': True,
            'cross_lag_beta': model.fe_params['user_c_lag_z'],
            'cross_lag_se': model.bse['user_c_lag_z'],
            'cross_lag_p': model.pvalues['user_c_lag_z'],
            'ar_beta': model.fe_params['ai_c_lag_z'],
            'n_obs': len(clean_df)
        }
    except Exception as e:
        return {'converged': False, 'error': str(e)}


# =============================================================================
# SINGLE SIMULATION
# =============================================================================

def run_single_simulation(args):
    """Run a single simulation iteration."""
    effect_size, sim_id, n_conv, mean_exch, sd_exch = args
    
    # Generate data
    df = generate_crosslag_data(
        n_conv=n_conv,
        mean_exch=mean_exch,
        sd_exch=sd_exch,
        cross_lag_effect=effect_size,
        seed=sim_id * 1000 + int(effect_size * 100)
    )
    
    # Fit model
    result = fit_crosslag_model(df)
    
    if result['converged']:
        return {
            'effect_size': effect_size,
            'sim_id': sim_id,
            'converged': True,
            'estimated_beta': result['cross_lag_beta'],
            'se': result['cross_lag_se'],
            'p_value': result['cross_lag_p'],
            'significant': result['cross_lag_p'] < ALPHA,
            'n_obs': result['n_obs']
        }
    else:
        return {
            'effect_size': effect_size,
            'sim_id': sim_id,
            'converged': False,
            'estimated_beta': np.nan,
            'se': np.nan,
            'p_value': np.nan,
            'significant': False,
            'n_obs': np.nan
        }


# =============================================================================
# POWER ANALYSIS
# =============================================================================

def run_power_analysis(effect_sizes, n_simulations, n_conv, mean_exch, sd_exch,
                       use_parallel=True):
    """
    Run Monte Carlo power analysis for multiple effect sizes.
    
    Parameters:
    -----------
    effect_sizes : list
        List of effect sizes to test
    n_simulations : int
        Number of simulations per effect size
    n_conv : int
        Number of conversations
    mean_exch : float
        Mean exchanges per conversation
    sd_exch : float
        SD of exchanges
    use_parallel : bool
        Whether to use parallel processing
    
    Returns:
    --------
    pd.DataFrame with power analysis results
    """
    print(f"\n{'='*70}")
    print("MONTE CARLO POWER ANALYSIS FOR CROSS-LAGGED PANEL MODEL")
    print(f"{'='*70}")
    print(f"\nSimulation Parameters:")
    print(f"  N conversations: {n_conv}")
    print(f"  Mean exchanges/conv: {mean_exch}")
    print(f"  N simulations per effect: {n_simulations}")
    print(f"  Alpha level: {ALPHA}")
    print(f"  Effect sizes tested: {effect_sizes}")
    
    # Prepare arguments for all simulations
    all_args = []
    for effect_size in effect_sizes:
        for sim_id in range(n_simulations):
            all_args.append((effect_size, sim_id, n_conv, mean_exch, sd_exch))
    
    # Run simulations
    print(f"\nRunning {len(all_args)} total simulations...")
    
    if use_parallel:
        n_cores = max(1, cpu_count() - 1)
        print(f"Using {n_cores} CPU cores")
        with Pool(n_cores) as pool:
            results = list(tqdm(pool.imap(run_single_simulation, all_args), 
                               total=len(all_args)))
    else:
        results = []
        for args in tqdm(all_args):
            results.append(run_single_simulation(args))
    
    results_df = pd.DataFrame(results)
    
    return results_df


def calculate_power_summary(results_df):
    """Calculate power summary statistics from simulation results."""
    summary = []
    
    for effect_size in results_df['effect_size'].unique():
        subset = results_df[results_df['effect_size'] == effect_size]
        converged = subset[subset['converged']]
        
        n_total = len(subset)
        n_converged = len(converged)
        n_significant = converged['significant'].sum()
        
        power = n_significant / n_converged if n_converged > 0 else np.nan
        
        # Calculate confidence interval for power using Wilson score interval
        if n_converged > 0:
            z = 1.96
            p_hat = power
            n = n_converged
            denominator = 1 + z**2/n
            centre = p_hat + z**2/(2*n)
            margin = z * np.sqrt((p_hat*(1-p_hat) + z**2/(4*n))/n)
            power_ci_low = (centre - margin) / denominator
            power_ci_high = (centre + margin) / denominator
        else:
            power_ci_low = power_ci_high = np.nan
        
        summary.append({
            'effect_size': effect_size,
            'n_simulations': n_total,
            'n_converged': n_converged,
            'convergence_rate': n_converged / n_total,
            'n_significant': n_significant,
            'power': power,
            'power_ci_low': power_ci_low,
            'power_ci_high': power_ci_high,
            'mean_beta': converged['estimated_beta'].mean(),
            'sd_beta': converged['estimated_beta'].std(),
            'bias': converged['estimated_beta'].mean() - effect_size,
            'mean_se': converged['se'].mean(),
            'mean_n_obs': converged['n_obs'].mean()
        })
    
    return pd.DataFrame(summary)


def find_mdes(power_summary, target_power=0.80):
    """
    Find minimum detectable effect size (MDES) for target power.
    Uses linear interpolation.
    """
    # Sort by effect size
    df = power_summary.sort_values('effect_size')
    
    # Find where power crosses target
    for i in range(len(df) - 1):
        if df.iloc[i]['power'] < target_power <= df.iloc[i+1]['power']:
            # Linear interpolation
            x1, y1 = df.iloc[i]['effect_size'], df.iloc[i]['power']
            x2, y2 = df.iloc[i+1]['effect_size'], df.iloc[i+1]['power']
            mdes = x1 + (target_power - y1) * (x2 - x1) / (y2 - y1)
            return mdes
    
    # If power never reaches target
    if df['power'].max() < target_power:
        return f"> {df['effect_size'].max()}"
    else:
        return f"< {df['effect_size'].min()}"


# =============================================================================
# SAMPLE SIZE ANALYSIS
# =============================================================================

def sample_size_analysis(effect_size, target_power=0.80, n_simulations=500,
                         sample_sizes=[30, 50, 73, 100, 150, 200]):
    """
    Determine required sample size for target power at given effect size.
    """
    print(f"\n{'='*70}")
    print(f"SAMPLE SIZE ANALYSIS (Effect Size = {effect_size})")
    print(f"{'='*70}")
    
    results = []
    
    for n_conv in sample_sizes:
        print(f"\nTesting N = {n_conv} conversations...")
        
        # Run simulations
        sim_results = []
        for sim_id in tqdm(range(n_simulations)):
            result = run_single_simulation((effect_size, sim_id, n_conv, 
                                           MEAN_EXCHANGES, SD_EXCHANGES))
            sim_results.append(result)
        
        sim_df = pd.DataFrame(sim_results)
        converged = sim_df[sim_df['converged']]
        
        power = converged['significant'].mean() if len(converged) > 0 else 0
        
        results.append({
            'n_conversations': n_conv,
            'effect_size': effect_size,
            'power': power,
            'n_converged': len(converged),
            'mean_beta': converged['estimated_beta'].mean()
        })
        
        print(f"  Power: {power:.3f}")
    
    return pd.DataFrame(results)


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run complete power analysis."""
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # =========================================================================
    # Part 1: Power Analysis for Different Effect Sizes
    # =========================================================================
    print("\n" + "="*70)
    print("PART 1: POWER ANALYSIS FOR OBSERVED SAMPLE SIZE")
    print("="*70)
    
    results_df = run_power_analysis(
        effect_sizes=EFFECT_SIZES,
        n_simulations=N_SIMULATIONS,
        n_conv=N_CONVERSATIONS,
        mean_exch=MEAN_EXCHANGES,
        sd_exch=SD_EXCHANGES,
        use_parallel=True
    )
    
    # Save raw results
    results_file = os.path.join(OUTPUT_DIR, f"power_raw_{TIMESTAMP}.csv")
    results_df.to_csv(results_file, index=False)
    print(f"\nRaw results saved: {results_file}")
    
    # Calculate summary
    power_summary = calculate_power_summary(results_df)
    
    # Find MDES
    mdes = find_mdes(power_summary, TARGET_POWER)
    
    # =========================================================================
    # Print Results
    # =========================================================================
    print("\n" + "="*70)
    print("POWER ANALYSIS RESULTS")
    print("="*70)
    print(f"\nSample: N = {N_CONVERSATIONS} conversations, ~{MEAN_EXCHANGES:.1f} exchanges each")
    print(f"Model: AI_t ~ AI_{{t-1}} + User_{{t-1}} + (1|conversation)")
    print(f"Alpha: {ALPHA}")
    print(f"Simulations per effect size: {N_SIMULATIONS}")
    
    print("\n" + "-"*70)
    print(f"{'Effect':>8} {'Power':>8} {'95% CI':>16} {'Mean β̂':>10} {'Bias':>8} {'Conv%':>8}")
    print("-"*70)
    
    for _, row in power_summary.iterrows():
        ci_str = f"[{row['power_ci_low']:.2f}, {row['power_ci_high']:.2f}]"
        print(f"{row['effect_size']:>8.2f} {row['power']:>8.3f} {ci_str:>16} "
              f"{row['mean_beta']:>10.4f} {row['bias']:>8.4f} {row['convergence_rate']*100:>7.1f}%")
    
    print("-"*70)
    print(f"\nMinimum Detectable Effect Size (MDES) for {TARGET_POWER*100:.0f}% power: β* = {mdes}")
    
    # =========================================================================
    # Part 2: Contextualize with Observed Effects
    # =========================================================================
    print("\n" + "="*70)
    print("OBSERVED EFFECTS vs. DETECTABLE EFFECTS")
    print("="*70)
    
    observed_effects = {
        'U→AI Immediate (RoBERTa)': 0.216,
        'U→AI Cross-Lag (RoBERTa)': 0.028,
        'AI→U Immediate (RoBERTa)': -0.042,
        'AI→U Cross-Lag (RoBERTa)': -0.010,
        'Human C→T Immediate': 0.086,
        'Human C→T Cross-Lag': 0.139,
    }
    
    print(f"\n{'Effect':.<40} {'β*':>8} {'Detectable?':>12}")
    print("-"*60)
    
    for name, beta in observed_effects.items():
        # Find power for this effect size
        closest_idx = (power_summary['effect_size'] - abs(beta)).abs().idxmin()
        closest_power = power_summary.loc[closest_idx, 'power']
        detectable = "Yes" if closest_power >= 0.80 else f"No ({closest_power:.0%})"
        print(f"{name:.<40} {beta:>8.3f} {detectable:>12}")
    
    # =========================================================================
    # Save Summary
    # =========================================================================
    summary_file = os.path.join(OUTPUT_DIR, f"power_summary_{TIMESTAMP}.csv")
    power_summary.to_csv(summary_file, index=False)
    print(f"\nSummary saved: {summary_file}")
    
    # Write text report
    report_file = os.path.join(OUTPUT_DIR, f"power_report_{TIMESTAMP}.txt")
    with open(report_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("MONTE CARLO POWER ANALYSIS REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*70 + "\n\n")
        
        f.write("STUDY PARAMETERS\n")
        f.write("-"*70 + "\n")
        f.write(f"N conversations: {N_CONVERSATIONS}\n")
        f.write(f"Mean exchanges per conversation: {MEAN_EXCHANGES}\n")
        f.write(f"Model: RI-CLPM (Mixed-effects with random intercepts)\n")
        f.write(f"Alpha level: {ALPHA}\n")
        f.write(f"N simulations per effect size: {N_SIMULATIONS}\n\n")
        
        f.write("POWER ANALYSIS RESULTS\n")
        f.write("-"*70 + "\n")
        f.write(f"{'Effect Size':<12} {'Power':<10} {'95% CI':<20}\n")
        f.write("-"*70 + "\n")
        for _, row in power_summary.iterrows():
            ci = f"[{row['power_ci_low']:.3f}, {row['power_ci_high']:.3f}]"
            f.write(f"{row['effect_size']:<12.2f} {row['power']:<10.3f} {ci:<20}\n")
        
        f.write(f"\nMDES for {TARGET_POWER*100:.0f}% power: β* = {mdes}\n\n")
        
        f.write("INTERPRETATION\n")
        f.write("-"*70 + "\n")
        f.write(f"With N={N_CONVERSATIONS} conversations and ~{MEAN_EXCHANGES:.0f} exchanges each,\n")
        f.write(f"this study has {TARGET_POWER*100:.0f}% power to detect effects of β* ≥ {mdes}\n\n")

    print(f"Report saved: {report_file}")
    
    return power_summary, results_df


if __name__ == "__main__":
    power_summary, results_df = main()

