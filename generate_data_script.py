#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate sample NPO healthcare data for analysis
This script creates realistic synthetic data modeling nonprofit hospital financials
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime

# Hospital profiles for realistic data generation
HOSPITAL_PROFILES = {
    'large_general': {
        'revenue_range': (150_000_000, 500_000_000),
        'charity_mean': 0.08, 'charity_std': 0.02,
        'admin_mean': 0.25, 'admin_std': 0.03,
        'community_mean': 0.05, 'community_std': 0.01,
        'growth_rate': 0.04
    },
    'medium_general': {
        'revenue_range': (50_000_000, 150_000_000),
        'charity_mean': 0.09, 'charity_std': 0.025,
        'admin_mean': 0.27, 'admin_std': 0.04,
        'community_mean': 0.04, 'community_std': 0.015,
        'growth_rate': 0.035
    },
    'small_general': {
        'revenue_range': (20_000_000, 50_000_000),
        'charity_mean': 0.07, 'charity_std': 0.02,
        'admin_mean': 0.30, 'admin_std': 0.05,
        'community_mean': 0.03, 'community_std': 0.01,
        'growth_rate': 0.02
    },
    'specialty': {
        'revenue_range': (30_000_000, 100_000_000),
        'charity_mean': 0.06, 'charity_std': 0.015,
        'admin_mean': 0.28, 'admin_std': 0.03,
        'community_mean': 0.025, 'community_std': 0.01,
        'growth_rate': 0.05
    }
}

# Hospital name templates
HOSPITAL_NAMES = {
    'large_general': [
        'St. Mary Medical Center',
        'University Hospital System',
        'Regional Medical Center',
        'Metropolitan Health Network'
    ],
    'medium_general': [
        'WestCare Medical Center',
        'Valley General Hospital',
        'Community Medical Center',
        'Riverside Hospital'
    ],
    'small_general': [
        'County General Hospital',
        'Memorial Hospital',
        'District Medical Center',
        'Rural Health System'
    ],
    'specialty': [
        'Regional Specialty Hospital',
        'Children\'s Medical Center',
        'Heart & Vascular Institute',
        'Cancer Treatment Center'
    ]
}

def generate_hospital_data(name, profile_type, start_year, num_years, seed=None):
    """Generate financial data for a single hospital over multiple years"""
    if seed is not None:
        np.random.seed(seed)
    
    profile = HOSPITAL_PROFILES[profile_type]
    
    # Base revenue
    base_revenue = np.random.uniform(*profile['revenue_range'])
    
    data = []
    for year_offset in range(num_years):
        year = start_year + year_offset
        
        # Calculate revenue with growth and random variation
        growth_factor = (1 + profile['growth_rate']) ** year_offset
        random_factor = 1 + np.random.normal(0, 0.02)
        revenue = base_revenue * growth_factor * random_factor
        
        # Calculate expense ratios with year-to-year variation
        charity_pct = np.random.normal(profile['charity_mean'], profile['charity_std'])
        admin_pct = np.random.normal(profile['admin_mean'], profile['admin_std'])
        community_pct = np.random.normal(profile['community_mean'], profile['community_std'])
        
        # Apply bounds to ensure realistic values
        charity_pct = np.clip(charity_pct, 0.02, 0.15)
        admin_pct = np.clip(admin_pct, 0.15, 0.40)
        community_pct = np.clip(community_pct, 0.01, 0.08)
        
        # Create record
        data.append({
            'year': year,
            'organization': name,
            'hospital_type': 'Specialty' if 'specialty' in profile_type else 'General',
            'total_revenue': round(revenue, 2),
            'charity_care': round(revenue * charity_pct, 2),
            'admin_expense': round(revenue * admin_pct, 2),
            'community_benefit': round(revenue * community_pct, 2),
            'operating_expense': round(revenue * 0.85, 2),  # Additional metric
            'net_income': round(revenue * np.random.uniform(0.02, 0.08), 2)  # 2-8% margin
        })
    
    return data

def generate_dataset(num_hospitals_per_type=2, start_year=2020, num_years=4, seed=42):
    """Generate complete dataset with multiple hospitals"""
    np.random.seed(seed)
    
    all_data = []
    hospital_id = 0
    
    for profile_type, names in HOSPITAL_NAMES.items():
        # Select random hospitals from each category
        selected_names = np.random.choice(names, 
                                        size=min(num_hospitals_per_type, len(names)), 
                                        replace=False)
        
        for name in selected_names:
            # Generate data with unique seed per hospital
            hospital_data = generate_hospital_data(
                name, profile_type, start_year, num_years, 
                seed=seed + hospital_id if seed else None
            )
            all_data.extend(hospital_data)
            hospital_id += 1
    
    return pd.DataFrame(all_data)

def add_derived_metrics(df):
    """Add additional calculated metrics to the dataset"""
    df = df.copy()
    
    # Total community investment (charity + community benefit)
    df['total_community_investment'] = df['charity_care'] + df['community_benefit']
    
    # Efficiency ratio
    df['efficiency_ratio'] = df['operating_expense'] / df['total_revenue']
    
    # Net margin
    df['net_margin'] = df['net_income'] / df['total_revenue']
    
    return df

def create_summary_statistics(df):
    """Generate summary statistics for the dataset"""
    print("\n=== Dataset Summary ===")
    print(f"Total records: {len(df)}")
    print(f"Years covered: {df['year'].min()} - {df['year'].max()}")
    print(f"Number of organizations: {df['organization'].nunique()}")
    print(f"\nRevenue range: ${df['total_revenue'].min():,.0f} - ${df['total_revenue'].max():,.0f}")
    print(f"Average revenue: ${df['total_revenue'].mean():,.0f}")
    
    print("\n=== Financial Ratios (% of revenue) ===")
    print(f"Charity care: {(df['charity_care'] / df['total_revenue']).mean()*100:.1f}% "
          f"(±{(df['charity_care'] / df['total_revenue']).std()*100:.1f}%)")
    print(f"Admin expense: {(df['admin_expense'] / df['total_revenue']).mean()*100:.1f}% "
          f"(±{(df['admin_expense'] / df['total_revenue']).std()*100:.1f}%)")
    print(f"Community benefit: {(df['community_benefit'] / df['total_revenue']).mean()*100:.1f}% "
          f"(±{(df['community_benefit'] / df['total_revenue']).std()*100:.1f}%)")

def main():
    parser = argparse.ArgumentParser(description='Generate sample NPO healthcare data')
    parser.add_argument('--hospitals', type=int, default=2,
                       help='Number of hospitals per type (default: 2)')
    parser.add_argument('--years', type=int, default=4,
                       help='Number of years to generate (default: 4)')
    parser.add_argument('--start-year', type=int, default=2020,
                       help='Starting year (default: 2020)')
    parser.add_argument('--output', type=str, default='npo_healthcare_sample.csv',
                       help='Output filename (default: npo_healthcare_sample.csv)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--include-derived', action='store_true',
                       help='Include additional derived metrics')
    
    args = parser.parse_args()
    
    print("Generating NPO healthcare sample data...")
    print(f"Configuration: {args.hospitals} hospitals per type, "
          f"{args.years} years starting from {args.start_year}")
    
    # Generate base dataset
    df = generate_dataset(
        num_hospitals_per_type=args.hospitals,
        start_year=args.start_year,
        num_years=args.years,
        seed=args.seed
    )
    
    # Add derived metrics if requested
    if args.include_derived:
        df = add_derived_metrics(df)
        print("Added derived metrics to dataset")
    
    # Save to file
    output_path = Path(args.output)
    df.to_csv(output_path, index=False)
    print(f"\nDataset saved to: {output_path}")
    
    # Display summary
    create_summary_statistics(df)
    
    # Show sample records
    print("\n=== Sample Records ===")
    print(df.head())
    
    # Generate metadata file
    metadata = {
    'generated_date': datetime.now().isoformat(),
    'num_records': int(len(df)),  # transfer to int
    'num_hospitals': int(df['organization'].nunique()),  # transfer to int
    'years': [int(year) for year in df['year'].unique()],  # transfer to int
    'seed': args.seed,
    'includes_derived_metrics': args.include_derived
}
    
    metadata_path = output_path.with_suffix('.json')
    import json
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata saved to: {metadata_path}")

if __name__ == "__main__":
    main()