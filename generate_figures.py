#!/usr/bin/env python
"""
Generate figures for Manuscript 1 from CSV results.
Run this script after experiments are complete.

Usage:
    python generate_figures.py --results ./my_results --output ./figures
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9


def save_figure(fig, output_path, filename):
    """Save figure in both PDF and PNG formats."""
    pdf_path = os.path.join(output_path, f"{filename}.pdf")
    png_path = os.path.join(output_path, f"{filename}.png")
    
    fig.savefig(pdf_path, dpi=150, bbox_inches='tight', format='pdf')
    fig.savefig(png_path, dpi=150, bbox_inches='tight', format='png')
    
    print(f"   Saved: {filename}.pdf")
    print(f"   Saved: {filename}.png")
    return pdf_path, png_path


def create_accuracy_comparison(df, output_path):
    """Create accuracy comparison bar chart."""
    print("  Creating accuracy comparison figure...")
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Prepare data
    summary = df.groupby(['dataset', 'architecture'])['accuracy'].agg(['mean', 'std']).reset_index()
    
    datasets = summary['dataset'].unique()
    x = np.arange(len(datasets))
    width = 0.35
    
    semi_means = []
    semi_stds = []
    conc_means = []
    conc_stds = []
    
    for dataset in datasets:
        semi = summary[(summary['dataset'] == dataset) & (summary['architecture'] == 'SemiSymbolic')]
        conc = summary[(summary['dataset'] == dataset) & (summary['architecture'] == 'Concurrency')]
        
        semi_means.append(semi['mean'].values[0] if len(semi) > 0 else 0)
        semi_stds.append(semi['std'].values[0] if len(semi) > 0 else 0)
        conc_means.append(conc['mean'].values[0] if len(conc) > 0 else 0)
        conc_stds.append(conc['std'].values[0] if len(conc) > 0 else 0)
    
    # Plot bars with error bars
    bars1 = ax.bar(x - width/2, semi_means, width, label='SemiSymbolic', 
                    color='#2E86AB', edgecolor='black', yerr=semi_stds, 
                    capsize=5, error_kw={'linewidth': 1, 'ecolor': 'black'})
    bars2 = ax.bar(x + width/2, conc_means, width, label='Concurrency', 
                    color='#A23B72', edgecolor='black', yerr=conc_stds,
                    capsize=5, error_kw={'linewidth': 1, 'ecolor': 'black'})
    
    # Add value labels
    for i, (semi, conc) in enumerate(zip(semi_means, conc_means)):
        ax.text(x[i] - width/2, semi + 0.02, f'{semi:.3f}', ha='center', fontsize=9)
        ax.text(x[i] + width/2, conc + 0.02, f'{conc:.3f}', ha='center', fontsize=9)
    
    # Formatting
    ax.set_xlabel('Dataset', fontsize=11, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
    ax.set_title('Accuracy Comparison: SemiSymbolic vs Concurrency', fontweight='bold', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=15, ha='right')
    ax.set_ylim(0, 1.0)
    ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_figure(fig, output_path, 'accuracy_comparison')
    plt.close()


def create_contingency_analysis(df, output_path):
    """Create contingency analysis plot."""
    print("  Creating contingency analysis figure...")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate accuracy difference
    summary = df.groupby(['dataset', 'architecture'])['accuracy'].mean().unstack()
    summary['diff'] = summary['Concurrency'] - summary['SemiSymbolic']
    summary = summary.reset_index()
    
    # Get contingency factors
    contingency = df[['dataset', 'dependency', 'error_tolerance', 'ambiguity']].drop_duplicates()
    summary = summary.merge(contingency, on='dataset')
    
    # Sort by diff
    summary = summary.sort_values('diff', ascending=True)
    
    # Plot
    colors = ['#2E86AB' if d < 0 else '#A23B72' for d in summary['diff']]
    bars = ax.barh(summary['dataset'], summary['diff'], color=colors, edgecolor='black', height=0.6)
    
    # Add value labels
    for i, (diff, dataset) in enumerate(zip(summary['diff'], summary['dataset'])):
        ax.text(diff + 0.01, i, f'{diff:+.3f}', 
                va='center', fontsize=10, fontweight='bold')
    
    # Add contingency information as annotations
    for i, row in summary.iterrows():
        ax.annotate(f'δ={row["dependency"]:.1f}, α={row["ambiguity"]:.1f}', 
                    xy=(0.02, i), xycoords='axes fraction',
                    fontsize=8, style='italic', ha='left')
    
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.set_xlabel('Accuracy Difference (Concurrency - SemiSymbolic)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Dataset', fontsize=11, fontweight='bold')
    ax.set_title('Contingency Analysis: Architecture Performance Difference', fontweight='bold', fontsize=12)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#A23B72', label='Concurrency wins'),
                       Patch(facecolor='#2E86AB', label='SemiSymbolic wins')]
    ax.legend(handles=legend_elements, loc='lower right', frameon=True, fancybox=True, shadow=True)
    
    plt.tight_layout()
    save_figure(fig, output_path, 'contingency_analysis')
    plt.close()


def create_latency_comparison(df, output_path):
    """Create latency comparison bar chart."""
    print("  Creating latency comparison figure...")
    fig, ax = plt.subplots(figsize=(8, 5))
    
    summary = df.groupby(['dataset', 'architecture'])['latency_ms'].agg(['mean', 'std']).reset_index()
    
    datasets = summary['dataset'].unique()
    x = np.arange(len(datasets))
    width = 0.35
    
    semi_means = []
    semi_stds = []
    conc_means = []
    conc_stds = []
    
    for dataset in datasets:
        semi = summary[(summary['dataset'] == dataset) & (summary['architecture'] == 'SemiSymbolic')]
        conc = summary[(summary['dataset'] == dataset) & (summary['architecture'] == 'Concurrency')]
        
        semi_means.append(semi['mean'].values[0] if len(semi) > 0 else 0)
        semi_stds.append(semi['std'].values[0] if len(semi) > 0 else 0)
        conc_means.append(conc['mean'].values[0] if len(conc) > 0 else 0)
        conc_stds.append(conc['std'].values[0] if len(conc) > 0 else 0)
    
    # Plot bars with error bars
    ax.bar(x - width/2, semi_means, width, label='SemiSymbolic', 
           color='#2E86AB', edgecolor='black', yerr=semi_stds,
           capsize=5, error_kw={'linewidth': 1, 'ecolor': 'black'})
    ax.bar(x + width/2, conc_means, width, label='Concurrency', 
           color='#A23B72', edgecolor='black', yerr=conc_stds,
           capsize=5, error_kw={'linewidth': 1, 'ecolor': 'black'})
    
    # Add value labels
    for i, (semi, conc) in enumerate(zip(semi_means, conc_means)):
        ax.text(x[i] - width/2, semi + 5, f'{semi:.1f}ms', ha='center', fontsize=9)
        ax.text(x[i] + width/2, conc + 5, f'{conc:.1f}ms', ha='center', fontsize=9)
    
    ax.set_xlabel('Dataset', fontsize=11, fontweight='bold')
    ax.set_ylabel('Latency (ms)', fontsize=11, fontweight='bold')
    ax.set_title('Latency Comparison: SemiSymbolic vs Concurrency', fontweight='bold', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=15, ha='right')
    ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    save_figure(fig, output_path, 'latency_comparison')
    plt.close()


def create_explainability_comparison(df, output_path):
    """Create explainability comparison bar chart."""
    print("  Creating explainability comparison figure...")
    fig, ax = plt.subplots(figsize=(8, 5))
    
    summary = df.groupby(['dataset', 'architecture'])['explainability_score'].mean().reset_index()
    
    datasets = summary['dataset'].unique()
    x = np.arange(len(datasets))
    width = 0.35
    
    semi_scores = []
    conc_scores = []
    
    for dataset in datasets:
        semi = summary[(summary['dataset'] == dataset) & (summary['architecture'] == 'SemiSymbolic')]
        conc = summary[(summary['dataset'] == dataset) & (summary['architecture'] == 'Concurrency')]
        
        semi_scores.append(semi['explainability_score'].values[0] if len(semi) > 0 else 0)
        conc_scores.append(conc['explainability_score'].values[0] if len(conc) > 0 else 0)
    
    ax.bar(x - width/2, semi_scores, width, label='SemiSymbolic', 
           color='#2E86AB', edgecolor='black')
    ax.bar(x + width/2, conc_scores, width, label='Concurrency', 
           color='#A23B72', edgecolor='black')
    
    # Add value labels
    for i, (semi, conc) in enumerate(zip(semi_scores, conc_scores)):
        ax.text(x[i] - width/2, semi + 0.02, f'{semi:.2f}', ha='center', fontsize=9)
        ax.text(x[i] + width/2, conc + 0.02, f'{conc:.2f}', ha='center', fontsize=9)
    
    ax.set_xlabel('Dataset', fontsize=11, fontweight='bold')
    ax.set_ylabel('Explainability Score', fontsize=11, fontweight='bold')
    ax.set_title('Explainability Comparison: SemiSymbolic vs Concurrency', fontweight='bold', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=15, ha='right')
    ax.set_ylim(0, 1)
    ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    save_figure(fig, output_path, 'explainability_comparison')
    plt.close()


def create_failure_rate_comparison(df, output_path):
    """Create failure rate comparison bar chart."""
    print("  Creating failure rate comparison figure...")
    fig, ax = plt.subplots(figsize=(8, 5))
    
    summary = df.groupby(['dataset', 'architecture'])['failure_rate'].mean().reset_index()
    
    datasets = summary['dataset'].unique()
    x = np.arange(len(datasets))
    width = 0.35
    
    semi_rates = []
    conc_rates = []
    
    for dataset in datasets:
        semi = summary[(summary['dataset'] == dataset) & (summary['architecture'] == 'SemiSymbolic')]
        conc = summary[(summary['dataset'] == dataset) & (summary['architecture'] == 'Concurrency')]
        
        semi_rates.append(semi['failure_rate'].values[0] if len(semi) > 0 else 0)
        conc_rates.append(conc['failure_rate'].values[0] if len(conc) > 0 else 0)
    
    ax.bar(x - width/2, semi_rates, width, label='SemiSymbolic', 
           color='#2E86AB', edgecolor='black')
    ax.bar(x + width/2, conc_rates, width, label='Concurrency', 
           color='#A23B72', edgecolor='black')
    
    # Add value labels
    for i, (semi, conc) in enumerate(zip(semi_rates, conc_rates)):
        ax.text(x[i] - width/2, semi + 0.01, f'{semi:.3f}', ha='center', fontsize=9)
        ax.text(x[i] + width/2, conc + 0.01, f'{conc:.3f}', ha='center', fontsize=9)
    
    ax.set_xlabel('Dataset', fontsize=11, fontweight='bold')
    ax.set_ylabel('Failure Rate', fontsize=11, fontweight='bold')
    ax.set_title('Failure Rate Comparison: SemiSymbolic vs Concurrency', fontweight='bold', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=15, ha='right')
    ax.set_ylim(0, 0.5)
    ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    save_figure(fig, output_path, 'failure_rate_comparison')
    plt.close()


def create_confidence_comparison(df, output_path):
    """Create confidence comparison bar chart."""
    print("  Creating confidence comparison figure...")
    fig, ax = plt.subplots(figsize=(8, 5))
    
    summary = df.groupby(['dataset', 'architecture'])['avg_confidence'].mean().reset_index()
    
    datasets = summary['dataset'].unique()
    x = np.arange(len(datasets))
    width = 0.35
    
    semi_conf = []
    conc_conf = []
    
    for dataset in datasets:
        semi = summary[(summary['dataset'] == dataset) & (summary['architecture'] == 'SemiSymbolic')]
        conc = summary[(summary['dataset'] == dataset) & (summary['architecture'] == 'Concurrency')]
        
        semi_conf.append(semi['avg_confidence'].values[0] if len(semi) > 0 else 0)
        conc_conf.append(conc['avg_confidence'].values[0] if len(conc) > 0 else 0)
    
    ax.bar(x - width/2, semi_conf, width, label='SemiSymbolic', 
           color='#2E86AB', edgecolor='black')
    ax.bar(x + width/2, conc_conf, width, label='Concurrency', 
           color='#A23B72', edgecolor='black')
    
    # Add value labels
    for i, (semi, conc) in enumerate(zip(semi_conf, conc_conf)):
        ax.text(x[i] - width/2, semi + 0.02, f'{semi:.2f}', ha='center', fontsize=9)
        ax.text(x[i] + width/2, conc + 0.02, f'{conc:.2f}', ha='center', fontsize=9)
    
    ax.set_xlabel('Dataset', fontsize=11, fontweight='bold')
    ax.set_ylabel('Average Confidence', fontsize=11, fontweight='bold')
    ax.set_title('Confidence Comparison: SemiSymbolic vs Concurrency', fontweight='bold', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=15, ha='right')
    ax.set_ylim(0, 1)
    ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    save_figure(fig, output_path, 'confidence_comparison')
    plt.close()


def create_results_table(df, output_path):
    """Create results table and summary statistics."""
    print("  Creating results table...")
    
    summary = df.groupby(['dataset', 'architecture']).agg({
        'accuracy': ['mean', 'std'],
        'latency_ms': 'mean',
        'explainability_score': 'mean',
        'failure_rate': 'mean',
        'avg_confidence': 'mean'
    }).round(4)
    
    # Flatten columns
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    summary = summary.reset_index()
    
    # Save as CSV
    csv_path = os.path.join(output_path, 'results_summary.csv')
    summary.to_csv(csv_path, index=False)
    print(f"   Saved: results_summary.csv")
    
    # Create simple text table
    txt_path = os.path.join(output_path, 'results_table.txt')
    with open(txt_path, 'w') as f:
        f.write("=" * 100 + "\n")
        f.write("RESULTS SUMMARY\n")
        f.write("=" * 100 + "\n\n")
        f.write(summary.to_string(index=False))
    print(f"   Saved: results_table.txt")
    
    return summary


def main():
    parser = argparse.ArgumentParser(description='Generate figures for Manuscript 1')
    parser.add_argument('--results', type=str, default='./my_results',
                        help='Path to results CSV file')
    parser.add_argument('--output', type=str, default='./figures',
                        help='Output directory for figures')
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    # Fix: Correct path construction
    results_path = os.path.join(args.results, 'comparison_results.csv')
    
    # Also try relative to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    alt_path = os.path.join(script_dir, 'my_results', 'comparison_results.csv')
    
    df = None
    
    if os.path.exists(results_path):
        df = pd.read_csv(results_path)
        print(f"✅ Loaded results from {results_path}")
    elif os.path.exists(alt_path):
        df = pd.read_csv(alt_path)
        print(f"✅ Loaded results from {alt_path}")
    else:
        print(f"❌ Error: Results file not found.")
        print(f"   Tried: {results_path}")
        print(f"   Tried: {alt_path}")
        print("\n   Make sure you've run experiments first:")
        print("   python experiments/run_comparison.py --datasets heart_disease,nsl_kdd --runs 20 --output ./my_results")
        sys.exit(1)
    
    print(f"\n📊 Data loaded:")
    print(f"   Datasets: {list(df['dataset'].unique())}")
    print(f"   Architectures: {list(df['architecture'].unique())}")
    print(f"   Total rows: {len(df)}")
    
    # Generate all figures
    print("\n📈 Generating figures (PDF and PNG formats):")
    print("-" * 50)
    
    create_accuracy_comparison(df, args.output)
    create_contingency_analysis(df, args.output)
    create_latency_comparison(df, args.output)
    create_explainability_comparison(df, args.output)
    create_failure_rate_comparison(df, args.output)
    create_confidence_comparison(df, args.output)
    create_results_table(df, args.output)
    
    print("\n" + "=" * 50)
    print("✅ ALL FIGURES GENERATED SUCCESSFULLY!")
    print("=" * 50)
    print(f"\n📁 Figures saved to: {args.output}/")
    print("\nGenerated files:")
    print("   • accuracy_comparison.pdf/png")
    print("   • contingency_analysis.pdf/png")
    print("   • latency_comparison.pdf/png")
    print("   • explainability_comparison.pdf/png")
    print("   • failure_rate_comparison.pdf/png")
    print("   • confidence_comparison.pdf/png")
    print("   • results_summary.csv")
    print("   • results_table.txt")


if __name__ == "__main__":
    main()