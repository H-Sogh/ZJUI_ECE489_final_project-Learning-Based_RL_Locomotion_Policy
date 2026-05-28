#!/usr/bin/env python3
"""
Generate publication-quality plots for Quadruped RL Locomotion final report.
Usage: python generate_plots.py [--test]  # --test generates sample CSVs first
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import argparse
import numpy as np
from pathlib import Path

# Professional style settings
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 14,
    'legend.fontsize': 10, 'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'figure.dpi': 100, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'savefig.facecolor': 'white', 'font.family': 'sans-serif'
})

def generate_sample_csvs():
    """Create test CSV files for validation (run with --test flag)."""
    print("🧪 Generating sample CSV files for testing...")
    Path('rewards').mkdir(exist_ok=True)
    Path('errors').mkdir(exist_ok=True)
    Path('loss').mkdir(exist_ok=True)
    Path('termination').mkdir(exist_ok=True)
    
    steps = np.linspace(0, 1000, 100)
    for terrain in ['flat', 'rough']:
        base = 5 if terrain == 'flat' else 3
        noise = 0.5 if terrain == 'flat' else 1.2
        # Mean reward
        pd.DataFrame({'Step': steps, 'Value': base + 2*np.log1p(steps/100) + np.random.randn(100)*noise, 'Std': np.abs(np.random.randn(100))*0.3}).to_csv(f'rewards/{terrain}-mean_reward.csv', index=False)
        # Velocity tracking reward
        pd.DataFrame({'Step': steps, 'Value': 1.2 - 0.8*np.exp(-steps/200) + np.random.randn(100)*0.1}).to_csv(f'rewards/{terrain}-track_lin_vel_xy_exp.csv', index=False)
        # Feet air time
        val = 0.22 if terrain == 'flat' else 0.008
        pd.DataFrame({'Step': steps, 'Value': np.full(100, val) + np.random.randn(100)*0.02}).to_csv(f'rewards/{terrain}-feet_air_time.csv', index=False)
        # Errors
        err_base = 0.08 if terrain == 'flat' else 0.15
        pd.DataFrame({'Step': steps, 'Value': err_base + 0.03*np.sin(steps/50) + np.random.randn(100)*0.02}).to_csv(f'errors/{terrain}-error_vel_xy.csv', index=False)
        pd.DataFrame({'Step': steps, 'Value': 0.02 + np.random.randn(100)*0.005}).to_csv(f'errors/{terrain}-error_vel_yaw.csv', index=False)
        # Losses
        pd.DataFrame({'Step': steps, 'Value': 2.0*np.exp(-steps/300) + np.random.randn(100)*0.1}).to_csv(f'loss/{terrain}-value-loss.csv', index=False)
        pd.DataFrame({'Step': steps, 'Value': 0.3*np.exp(-steps/400) + np.random.randn(100)*0.05}).to_csv(f'loss/{terrain}-surrogate-loss.csv', index=False)
        pd.DataFrame({'Step': steps, 'Value': 0.01 + 0.005*np.exp(-steps/500) + np.random.randn(100)*0.002}).to_csv(f'loss/{terrain}-entropy-loss.csv', index=False)
        pd.DataFrame({'Step': steps, 'Value': 1e-3 * (0.5 + 0.5*np.cos(steps/200))}).to_csv(f'loss/{terrain}-learning_rate.csv', index=False)
        # Termination
        pd.DataFrame({'Step': range(20), 'Value': [0]*18 + [1, 1] if terrain=='rough' else [0]*20}).to_csv(f'termination/{terrain}-base_contact.csv', index=False)
        pd.DataFrame({'Step': range(20), 'Value': [1]*20}).to_csv(f'termination/{terrain}-time_out.csv', index=False)
    # Root level files
    pd.DataFrame({'Step': range(300), 'Value': 200 + 50*np.sin(np.linspace(0, 4*np.pi, 300)) + np.random.randn(300)*10}).to_csv('flat-mean_episode_length.csv', index=False)
    pd.DataFrame({'Step': range(1500), 'Value': 150 + 30*np.sin(np.linspace(0, 2*np.pi, 1500)) + np.random.randn(1500)*15}).to_csv('rough-mean_episode_length.csv', index=False)
    pd.DataFrame({'Step': range(1500), 'Value': np.minimum(10, np.linspace(0, 10, 1500))}).to_csv('rough-Curriculum-terrain-levels.csv', index=False)
    print("✓ Sample CSVs created. Run without --test to generate plots.\n")

def load_csv(filepath):
    """Load CSV with flexible column handling."""
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    col_map = {c: c.capitalize() if c.lower() in ['step', 'value', 'std'] else c for c in df.columns}
    df = df.rename(columns={k: v for k, v in col_map.items() if k.lower() in ['step', 'value', 'std']})
    if 'Step' not in df.columns:
        df['Step'] = range(len(df))
    return df

def plot_comparison(flat_path, rough_path, output_name, title, ylabel, xlabel='Step', subdir='summary'):
    """Generate flat vs rough comparison plot."""
    try:
        flat_df, rough_df = load_csv(flat_path), load_csv(rough_path)
    except FileNotFoundError as e:
        print(f"⚠ Skipping {output_name}: {e}")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
    ax.plot(flat_df['Step'], flat_df['Value'], label='Flat Terrain', linewidth=1.5, color='#1f77b4')
    ax.plot(rough_df['Step'], rough_df['Value'], label='Rough Terrain', linewidth=1.5, alpha=0.85, color='#ff7f0e')
    
    if 'Std' in flat_df.columns:
        ax.fill_between(flat_df['Step'], flat_df['Value']-flat_df['Std'], flat_df['Value']+flat_df['Std'], alpha=0.15, color='#1f77b4')
    if 'Std' in rough_df.columns:
        ax.fill_between(rough_df['Step'], rough_df['Value']-rough_df['Std'], rough_df['Value']+rough_df['Std'], alpha=0.15, color='#ff7f0e')
    
    ax.set_xlabel(xlabel, fontsize=12, labelpad=8)
    ax.set_ylabel(ylabel, fontsize=12, labelpad=8)
    ax.set_title(title, fontsize=14, pad=15, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.minorticks_on()
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.legend(loc='best', fontsize=10, framealpha=0.95)
    plt.tight_layout()
    
    Path(f'figures/{subdir}').mkdir(parents=True, exist_ok=True)
    for fmt in ['pdf', 'png']:
        plt.savefig(f'figures/{subdir}/{output_name}.{fmt}', format=fmt, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ figures/{subdir}/{output_name}.{{pdf,png}}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', help='Generate sample CSVs for testing')
    args = parser.parse_args()
    
    if args.test:
        generate_sample_csvs()
        return
    
    print("\n=== REWARD PLOTS ===")
    plot_comparison('rewards/flat-mean_reward.csv', 'rewards/rough-mean_reward.csv', 
                   'mean_reward_comparison', 'Mean Episode Reward', 'Reward Value')
    plot_comparison('rewards/flat-track_lin_vel_xy_exp.csv', 'rewards/rough-track_lin_vel_xy_exp.csv',
                   'track_lin_vel_xy_exp', 'Velocity Tracking Reward', 'Reward Contribution')
    plot_comparison('rewards/flat-feet_air_time.csv', 'rewards/rough-feet_air_time.csv',
                   'feet_air_time_reward', 'Foot Clearance Reward', 'Reward Contribution')
    
    print("\n=== ERROR PLOTS ===")
    plot_comparison('errors/flat-error_vel_xy.csv', 'errors/rough-error_vel_xy.csv',
                   'error_vel_xy', 'XY Velocity Tracking Error (RMS)', 'Error (m/s)', subdir='errors')
    plot_comparison('errors/flat-error_vel_yaw.csv', 'errors/rough-error_vel_yaw.csv',
                   'error_vel_yaw', 'Yaw Velocity Tracking Error', 'Error (rad/s)', subdir='errors')
    
    print("\n=== LOSS PLOTS ===")
    for name, title, ylabel in [
        ('value-loss', 'PPO Value Function Loss', 'Loss'),
        ('surrogate-loss', 'PPO Surrogate Loss (Clipped)', 'Loss'),
        ('entropy-loss', 'Policy Entropy (Exploration)', 'Entropy'),
        ('learning_rate', 'Adaptive Learning Rate', 'Learning Rate')
    ]:
        plot_comparison(f'loss/flat-{name}.csv', f'loss/rough-{name}.csv',
                       name.replace('-', '_'), title, ylabel, subdir='loss')
    
    print("\n=== TERMINATION PLOTS ===")
    plot_comparison('termination/flat-base_contact.csv', 'termination/rough-base_contact.csv',
                   'base_contact_termination', 'Base Contact Events', 'Count', 'Episode', subdir='termination')
    plot_comparison('termination/flat-time_out.csv', 'termination/rough-time_out.csv',
                   'timeout_success', 'Episode Completion Rate', 'Success Rate', 'Episode', subdir='termination')
    
    print("\n=== EPISODE METRICS ===")
    plot_comparison('flat-mean_episode_length.csv', 'rough-mean_episode_length.csv',
                   'episode_length', 'Mean Episode Length', 'Steps', 'Training Iteration')
    
    # Curriculum plot (rough only)
    try:
        curr = load_csv('rough-Curriculum-terrain-levels.csv')
        fig, ax = plt.subplots(figsize=(12, 5), dpi=100)
        ax.plot(curr['Step'], curr['Value'], linewidth=2, color='#2ca02c')
        ax.set_xlabel('Training Step', fontsize=12); ax.set_ylabel('Curriculum Level', fontsize=12)
        ax.set_title('Rough Terrain: Curriculum Progression', fontsize=14, fontweight='bold', pad=15)
        ax.grid(True, linestyle='--', alpha=0.5); ax.minorticks_on()
        ax.xaxis.set_minor_locator(AutoMinorLocator(5)); ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        plt.tight_layout()
        for fmt in ['pdf', 'png']:
            plt.savefig(f'figures/summary/curriculum_progression.{fmt}', format=fmt, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print("✓ figures/summary/curriculum_progression.{pdf,png}")
    except FileNotFoundError:
        print("⚠ Skipping curriculum plot: rough-Curriculum-terrain-levels.csv not found")
    
    print("\n" + "="*60 + "\n✅ All plots generated! Check figures/ directory.\n" + "="*60)

if __name__ == '__main__':
    main()