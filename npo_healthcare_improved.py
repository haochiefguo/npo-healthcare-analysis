#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NPO Healthcare Financial Analysis
Analyzing nonprofit hospital compliance with proposed BBB legislation

Data Format Requirements:
- year: int, fiscal year
- organization: str, hospital name
- total_revenue: float, total annual revenue in USD
- charity_care: float, uncompensated care provided in USD
- admin_expense: float, administrative expenses in USD
- community_benefit: float, community benefit spending in USD
- hospital_type: str, optional, 'General' or 'Specialty'
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuration
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.figsize': (10, 6),
    'font.size': 10,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'figure.dpi': 100
})

# Constants for BBB thresholds
BBB_MIN_CHARITY_PCT = 0.08  # 8% minimum charity care
BBB_MAX_ADMIN_PCT = 0.30    # 30% maximum admin expense

def load_and_validate_data(filepath):
    """Load NPO healthcare data with validation"""
    try:
        df = pd.read_csv(filepath)
        
        # Validate required columns
        required_cols = ['year', 'organization', 'total_revenue', 
                        'charity_care', 'admin_expense', 'community_benefit']
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")
        
        # Check for negative values
        numeric_cols = ['total_revenue', 'charity_care', 'admin_expense', 'community_benefit']
        if (df[numeric_cols] < 0).any().any():
            print("Warning: Negative values detected in financial data")
        
        print(f"Loaded {len(df)} records from {filepath}")
        return df
    except FileNotFoundError:
        print(f"\nError: Data file not found at {filepath}")
        print("Please run 'python generate_sample_data.py' first to create sample data.")
        raise
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

def calculate_financial_ratios(df):
    """Calculate key financial ratios with zero-division handling"""
    df = df.copy()
    
    # Safe division to handle zero revenue cases
    df['charity_pct'] = np.where(df['total_revenue'] > 0, 
                                 df['charity_care'] / df['total_revenue'], 
                                 0)
    df['admin_pct'] = np.where(df['total_revenue'] > 0,
                               df['admin_expense'] / df['total_revenue'],
                               0)
    df['community_pct'] = np.where(df['total_revenue'] > 0,
                                   df['community_benefit'] / df['total_revenue'],
                                   0)
    
    # Add BBB compliance flags
    df['meets_charity_min'] = df['charity_pct'] >= BBB_MIN_CHARITY_PCT
    df['meets_admin_max'] = df['admin_pct'] <= BBB_MAX_ADMIN_PCT
    df['bbb_compliant'] = df['meets_charity_min'] & df['meets_admin_max']
    
    return df

def create_revenue_trend_plot(df):
    """Plot revenue trends with confidence intervals"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Calculate mean and std by year
    revenue_stats = df.groupby('year')['total_revenue'].agg(['mean', 'std']).reset_index()
    
    # Main trend lines
    for org in df['organization'].unique():
        org_data = df[df['organization'] == org]
        ax.plot(org_data['year'], org_data['total_revenue'], 
                marker='o', linewidth=2, label=org, alpha=0.8)
    
    # Add confidence band for overall trend
    ax.fill_between(revenue_stats['year'],
                   revenue_stats['mean'] - revenue_stats['std'],
                   revenue_stats['mean'] + revenue_stats['std'],
                   alpha=0.2, color='gray', label='±1 SD')
    
    ax.set_xlabel('Year')
    ax.set_ylabel('Total Revenue (USD)')
    ax.set_title('Revenue Trends by Organization')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Format y-axis as currency
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    plt.tight_layout()
    return fig

def create_bbb_risk_matrix(df):
    """Create BBB compliance risk matrix with annotations"""
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Define colors
    compliant_color = '#2E86AB'  # Blue
    non_compliant_color = '#E63946'  # Red
    
    # Create scatter plot with different markers for compliance status
    for compliant in [True, False]:
        mask = df['bbb_compliant'] == compliant
        subset = df[mask]
        ax.scatter(subset['admin_pct'] * 100, 
                  subset['charity_pct'] * 100,
                  s=200,
                  alpha=0.7,
                  label='Compliant' if compliant else 'Non-compliant',
                  color=compliant_color if compliant else non_compliant_color,
                  marker='o' if compliant else 'D',
                  edgecolors='white',
                  linewidth=2)
    
    # Add threshold lines with better styling
    ax.axhline(BBB_MIN_CHARITY_PCT * 100, color='#E63946', linestyle='--', 
               linewidth=2.5, label=f'BBB Min Charity ({BBB_MIN_CHARITY_PCT*100}%)', alpha=0.8)
    ax.axvline(BBB_MAX_ADMIN_PCT * 100, color='#F77F00', linestyle='--', 
               linewidth=2.5, label=f'BBB Max Admin ({BBB_MAX_ADMIN_PCT*100}%)', alpha=0.8)
    
    # Shade risk zones with lighter colors
    ax.axhspan(0, BBB_MIN_CHARITY_PCT * 100, 
               xmax=1, alpha=0.15, color='#E63946', label='Charity Care Risk Zone')
    ax.axvspan(BBB_MAX_ADMIN_PCT * 100, 100, 
               ymax=1, alpha=0.15, color='#F77F00', label='Admin Cost Risk Zone')
    
    # Smart annotation - only label organizations at highest risk
    # Calculate risk score (distance from safe zone)
    df_risk = df.copy()
    df_risk['charity_gap'] = np.maximum(0, BBB_MIN_CHARITY_PCT - df_risk['charity_pct'])
    df_risk['admin_gap'] = np.maximum(0, df_risk['admin_pct'] - BBB_MAX_ADMIN_PCT)
    df_risk['risk_score'] = np.sqrt(df_risk['charity_gap']**2 + df_risk['admin_gap']**2)
    
    # Get the highest risk organizations (latest year for each org)
    latest_year = df_risk['year'].max()
    high_risk = df_risk[~df_risk['bbb_compliant']]
    
    # Group by organization and get the latest year data
    high_risk_latest = high_risk.loc[high_risk.groupby('organization')['year'].idxmax()]
    high_risk_latest = high_risk_latest.nlargest(4, 'risk_score')
    
    # Annotate with smart positioning to avoid overlap
    positions = []
    for idx, row in high_risk_latest.iterrows():
        x, y = row['admin_pct'] * 100, row['charity_pct'] * 100
        
        # Calculate offset based on existing annotations
        if len(positions) == 0:
            xytext = (10, 10)
        else:
            # Check distances to existing annotations
            min_dist = float('inf')
            best_offset = (10, 10)
            
            # Try different offset positions
            for offset in [(10, 10), (-80, 10), (10, -30), (-80, -30)]:
                test_x = x + offset[0]/5
                test_y = y + offset[1]/5
                
                # Calculate min distance to existing annotations
                dist = min([np.sqrt((test_x - px)**2 + (test_y - py)**2) 
                           for px, py in positions] + [float('inf')])
                
                if dist > min_dist:
                    min_dist = dist
                    best_offset = offset
            
            xytext = best_offset
        
        positions.append((x + xytext[0]/5, y + xytext[1]/5))
        
        ax.annotate(f"{row['organization']}\n{row['year']}",
                   (x, y),
                   xytext=xytext, 
                   textcoords='offset points',
                   fontsize=9,
                   bbox=dict(boxstyle='round,pad=0.5', 
                            facecolor='white', 
                            edgecolor=non_compliant_color,
                            alpha=0.9),
                   arrowprops=dict(arrowstyle='->', 
                                 connectionstyle='arc3,rad=0.3',
                                 color=non_compliant_color,
                                 alpha=0.6))
    
    # Styling improvements
    ax.set_xlabel('Administrative Expenses (% of Revenue)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Charity Care (% of Revenue)', fontsize=14, fontweight='bold')
    ax.set_title('BBB Compliance Risk Matrix: Charity vs Administrative Spending', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Improve legend
    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, fontsize=11)
    
    # Grid styling
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=1)
    ax.set_axisbelow(True)
    
    # Set reasonable axis limits
    ax.set_xlim(15, max(45, df['admin_pct'].max() * 110))
    ax.set_ylim(0, max(15, df['charity_pct'].max() * 110))
    
    # Add text to explain zones
    ax.text(17, 1, 'HIGH RISK\nZONE', fontsize=12, color='#E63946', 
            alpha=0.7, fontweight='bold', ha='left')
    
    # Style the plot area
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    
    plt.tight_layout()
    return fig

def generate_compliance_summary(df):
    """Generate compliance summary statistics"""
    summary = {
        'total_observations': len(df),
        'compliant_count': df['bbb_compliant'].sum(),
        'compliance_rate': df['bbb_compliant'].mean() * 100,
        'avg_charity_pct': df['charity_pct'].mean() * 100,
        'avg_admin_pct': df['admin_pct'].mean() * 100,
        'orgs_below_charity_min': (~df['meets_charity_min']).sum(),
        'orgs_above_admin_max': (~df['meets_admin_max']).sum()
    }
    
    print("\n=== BBB Compliance Summary ===")
    print(f"Total observations: {summary['total_observations']}")
    print(f"Compliant: {summary['compliant_count']} ({summary['compliance_rate']:.1f}%)")
    print(f"Average charity care: {summary['avg_charity_pct']:.1f}%")
    print(f"Average admin expense: {summary['avg_admin_pct']:.1f}%")
    print(f"Below charity minimum: {summary['orgs_below_charity_min']}")
    print(f"Above admin maximum: {summary['orgs_above_admin_max']}")
    
    return summary

def main():
    """Main analysis pipeline"""
    # Update this path to your data file
    data_path = Path("npo_healthcare_sample.csv")  # Uses local directory by default
    
    print("Loading NPO healthcare data...")
    df = load_and_validate_data(data_path)
    
    print("Calculating financial ratios...")
    df = calculate_financial_ratios(df)
    
    # Add hospital type if not present
    if 'hospital_type' not in df.columns:
        # Infer from organization name or assign default
        df['hospital_type'] = df['organization'].apply(
            lambda x: 'Specialty' if 'Specialty' in x else 'General'
        )
    
    print("\nGenerating visualizations...")
    
    # 1. Revenue trends
    fig1 = create_revenue_trend_plot(df)
    plt.show()
    
    # 2. BBB risk matrix
    fig2 = create_bbb_risk_matrix(df)
    plt.show()
    
    # 3. Compliance summary
    summary = generate_compliance_summary(df)
    
    # 4. Year-over-year analysis
    yoy_changes = df.groupby('organization').apply(
        lambda x: x.sort_values('year')['charity_pct'].pct_change().mean() * 100
    )
    
    print("\n=== Year-over-Year Charity % Changes ===")
    for org, change in yoy_changes.items():
        direction = "↑" if change > 0 else "↓"
        print(f"{org}: {change:+.1f}% {direction}")
    
    # Save processed data
    output_path = data_path.parent / 'npo_healthcare_analysis_results.csv'
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

if __name__ == "__main__":
    main()