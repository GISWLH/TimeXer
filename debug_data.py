#!/usr/bin/env python3
"""
Debug script to examine the geo.csv data and diagnose temperature prediction issues
"""

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

def examine_geo_data():
    """Examine the geo.csv data structure and values"""
    print("="*60)
    print("GEO.CSV DATA EXAMINATION")
    print("="*60)
    
    # Load the data
    geo_url = "data/geo.csv"
    geo_df = pd.read_csv(geo_url)
    
    print(f"Dataset shape: {geo_df.shape}")
    print(f"Columns: {list(geo_df.columns)}")
    
    # Check first few rows
    print("\nFirst 5 rows:")
    print(geo_df.head())
    
    # Check data types
    print("\nData types:")
    print(geo_df.dtypes)
    
    # Check for missing values
    print("\nMissing values:")
    print(geo_df.isnull().sum())
    
    # Focus on temperature_2m
    print("\n" + "="*60)
    print("TEMPERATURE_2M ANALYSIS")
    print("="*60)
    
    temp_col = 'temperature_2m'
    if temp_col in geo_df.columns:
        temp_data = geo_df[temp_col]
        print(f"Temperature statistics:")
        print(temp_data.describe())
        
        print(f"\nTemperature range:")
        print(f"Min: {temp_data.min():.4f}")
        print(f"Max: {temp_data.max():.4f}")
        print(f"Mean: {temp_data.mean():.4f}")
        
        print(f"\nFirst 10 temperature values:")
        print(temp_data.head(10).tolist())
        
        print(f"\nLast 10 temperature values:")
        print(temp_data.tail(10).tolist())
        
        # Check if temperatures are in Kelvin (should be around 250-320K)
        if temp_data.min() > 200 and temp_data.max() < 400:
            print("\n✓ Temperature values appear to be in Kelvin (reasonable range)")
            print(f"  In Celsius: {temp_data.min()-273.15:.1f}°C to {temp_data.max()-273.15:.1f}°C")
        else:
            print("\n⚠ Temperature values may not be in Kelvin")
            
        # Check for zero or very small values
        zero_count = (temp_data == 0).sum()
        near_zero_count = (temp_data < 1).sum()
        print(f"\nZero values: {zero_count}")
        print(f"Near-zero values (< 1): {near_zero_count}")
        
    else:
        print(f"❌ Column '{temp_col}' not found!")
        
    return geo_df

def test_preprocessing():
    """Test the data preprocessing steps"""
    print("\n" + "="*60)
    print("PREPROCESSING TEST")
    print("="*60)
    
    # Load data
    geo_df = pd.read_csv("data/geo.csv")
    
    # Preprocess the data (same as notebook)
    geo_df['time'] = geo_df['time'].astype(str)
    geo_df['year'] = geo_df['time'].str[:4].astype(int)
    geo_df['month'] = geo_df['time'].str[4:].astype(int)
    geo_df['ds'] = pd.to_datetime(geo_df[['year', 'month']].assign(day=1))
    
    # Add unique_id
    geo_df['unique_id'] = 'GEO'
    
    # Rename temperature column to 'y'
    if 'temperature_2m' in geo_df.columns:
        geo_df = geo_df.rename(columns={'temperature_2m': 'y'})
        
        # Select features
        feature_columns = [
            'surface_pressure',
            'u_component_of_wind_10m',
            'v_component_of_wind_10m',
            'surface_net_solar_radiation_sum',
            'total_precipitation_sum',
            'total_evaporation_sum'
        ]
        
        # Create final dataset
        geo_final = geo_df[['unique_id', 'ds', 'y'] + feature_columns].copy()
        geo_final = geo_final.dropna()
        
        print(f"Final dataset shape: {geo_final.shape}")
        print(f"\nTarget variable (y) statistics after preprocessing:")
        print(geo_final['y'].describe())
        
        print(f"\nFirst 5 rows of processed data:")
        print(geo_final.head())
        
        print(f"\nTarget values (first 10):")
        print(geo_final['y'].head(10).tolist())
        
        # Check if there are any issues with the target variable
        if geo_final['y'].std() == 0:
            print("❌ All target values are the same!")
        elif geo_final['y'].min() == geo_final['y'].max():
            print("❌ No variation in target values!")
        else:
            print("✓ Target variable has variation")
            
        return geo_final
    else:
        print("❌ temperature_2m column not found after preprocessing")
        return None

def create_simple_visualization(geo_final):
    """Create simple visualizations to check data"""
    if geo_final is None:
        return
        
    print("\n" + "="*60)
    print("CREATING DIAGNOSTIC PLOTS")
    print("="*60)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Data Diagnostic Plots', fontsize=14)
    
    # 1. Temperature time series
    ax1 = axes[0, 0]
    ax1.plot(geo_final['ds'], geo_final['y'] - 273.15, marker='o', alpha=0.7)
    ax1.set_title('Temperature Time Series')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Temperature (°C)')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. Temperature distribution
    ax2 = axes[0, 1]
    temp_celsius = geo_final['y'] - 273.15
    ax2.hist(temp_celsius, bins=30, alpha=0.7, edgecolor='black')
    ax2.set_title('Temperature Distribution')
    ax2.set_xlabel('Temperature (°C)')
    ax2.set_ylabel('Frequency')
    ax2.grid(True, alpha=0.3)
    
    # 3. Check for seasonal patterns
    ax3 = axes[1, 0]
    geo_final['month'] = geo_final['ds'].dt.month
    monthly_temp = geo_final.groupby('month')['y'].mean() - 273.15
    ax3.plot(monthly_temp.index, monthly_temp.values, marker='o')
    ax3.set_title('Average Temperature by Month')
    ax3.set_xlabel('Month')
    ax3.set_ylabel('Temperature (°C)')
    ax3.grid(True, alpha=0.3)
    
    # 4. Check one feature correlation
    ax4 = axes[1, 1]
    if 'surface_net_solar_radiation_sum' in geo_final.columns:
        ax4.scatter(geo_final['surface_net_solar_radiation_sum'], 
                   geo_final['y'] - 273.15, alpha=0.5)
        ax4.set_title('Temperature vs Solar Radiation')
        ax4.set_xlabel('Solar Radiation')
        ax4.set_ylabel('Temperature (°C)')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('data_diagnostic.png', dpi=150, bbox_inches='tight')
    print("✓ Diagnostic plots saved as 'data_diagnostic.png'")
    plt.show()

def main():
    """Main function to run all diagnostics"""
    print("Starting data diagnostic...")
    
    # Step 1: Examine raw data
    geo_df = examine_geo_data()
    
    # Step 2: Test preprocessing
    geo_final = test_preprocessing()
    
    # Step 3: Create visualizations
    create_simple_visualization(geo_final)
    
    print("\n" + "="*60)
    print("DIAGNOSTIC COMPLETE")
    print("="*60)
    
    if geo_final is not None:
        print(f"✓ Data loaded successfully")
        print(f"✓ Final dataset: {geo_final.shape[0]} samples, {geo_final.shape[1]} features")
        print(f"✓ Temperature range: {geo_final['y'].min():.1f}K to {geo_final['y'].max():.1f}K")
        print(f"  ({geo_final['y'].min()-273.15:.1f}°C to {geo_final['y'].max()-273.15:.1f}°C)")
    else:
        print("❌ Issues found with data processing")

if __name__ == "__main__":
    main()