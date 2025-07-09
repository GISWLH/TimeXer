#!/usr/bin/env python3
"""
Test script to debug the model prediction issues
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def test_minimal_model():
    """Test with minimal model setup to identify issues"""
    print("="*60)
    print("MINIMAL MODEL TEST")
    print("="*60)
    
    # Load and preprocess data (same as notebook)
    geo_df = pd.read_csv("data/geo.csv")
    
    # Preprocess
    geo_df['time'] = geo_df['time'].astype(str)
    geo_df['year'] = geo_df['time'].str[:4].astype(int)
    geo_df['month'] = geo_df['time'].str[4:].astype(int)
    geo_df['ds'] = pd.to_datetime(geo_df[['year', 'month']].assign(day=1))
    geo_df['unique_id'] = 'GEO'
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
    
    geo_final = geo_df[['unique_id', 'ds', 'y'] + feature_columns].copy()
    geo_final = geo_final.dropna()
    
    print(f"Data shape: {geo_final.shape}")
    print(f"Temperature range: {geo_final['y'].min():.1f}K to {geo_final['y'].max():.1f}K")
    
    # Check data requirements for NeuralForecast
    print(f"\nData requirements check:")
    print(f"- Minimum data points needed: 168 (INPUT_SIZE=36 months × 4 + HORIZON=12)")
    print(f"- Available data points: {len(geo_final)}")
    print(f"- Date range: {geo_final['ds'].min()} to {geo_final['ds'].max()}")
    
    # Check if we have enough data for cross-validation
    horizon = 12
    input_size = 36
    n_windows = 5
    min_required = input_size + horizon * n_windows
    
    print(f"- Required for CV (input_size + horizon × n_windows): {min_required}")
    print(f"- Sufficient data: {'✓' if len(geo_final) >= min_required else '❌'}")
    
    if len(geo_final) < min_required:
        print(f"⚠ Not enough data for current settings!")
        print(f"  Reducing parameters...")
        horizon = 6
        input_size = 24
        n_windows = 3
        min_required = input_size + horizon * n_windows
        print(f"  New requirements: {min_required} (horizon={horizon}, input_size={input_size}, n_windows={n_windows})")
    
    # Try a simple model
    try:
        from neuralforecast.core import NeuralForecast
        from neuralforecast.models import NHITS
        
        # Use only NHITS for testing (simpler model)
        simple_model = NHITS(
            h=horizon,
            input_size=input_size,
            futr_exog_list=feature_columns,
            max_steps=100  # Reduced for faster testing
        )
        
        print(f"\nTesting with simplified NHITS model...")
        print(f"- Horizon: {horizon}")
        print(f"- Input size: {input_size}")
        print(f"- Max steps: 100")
        
        # Set environment
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        
        nf = NeuralForecast(models=[simple_model], freq="MS")
        
        # Try cross-validation with reduced windows
        print(f"Starting cross-validation with {n_windows} windows...")
        cv_preds = nf.cross_validation(
            geo_final, 
            step_size=horizon, 
            n_windows=n_windows
        )
        
        print(f"✓ Cross-validation completed!")
        print(f"Predictions shape: {cv_preds.shape}")
        print(f"Columns: {list(cv_preds.columns)}")
        
        # Check the predictions
        print(f"\nPrediction analysis:")
        print(f"Actual values (y):")
        print(f"  Mean: {cv_preds['y'].mean():.4f}K ({cv_preds['y'].mean()-273.15:.1f}°C)")
        print(f"  Std: {cv_preds['y'].std():.4f}K")
        print(f"  Range: {cv_preds['y'].min():.1f}K to {cv_preds['y'].max():.1f}K")
        print(f"  First 5 values: {cv_preds['y'].head().tolist()}")
        
        print(f"\nNHITS predictions:")
        nhits_col = 'NHITS'
        if nhits_col in cv_preds.columns:
            print(f"  Mean: {cv_preds[nhits_col].mean():.4f}K ({cv_preds[nhits_col].mean()-273.15:.1f}°C)")
            print(f"  Std: {cv_preds[nhits_col].std():.4f}K")
            print(f"  Range: {cv_preds[nhits_col].min():.1f}K to {cv_preds[nhits_col].max():.1f}K")
            print(f"  First 5 values: {cv_preds[nhits_col].head().tolist()}")
            
            # Calculate simple metrics
            mae = np.mean(np.abs(cv_preds['y'] - cv_preds[nhits_col]))
            print(f"\nSimple MAE: {mae:.4f}K ({mae:.2f}°C)")
            
        return cv_preds
        
    except Exception as e:
        print(f"❌ Error in model testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def create_simple_prediction_plot(cv_preds):
    """Create simple plots to visualize predictions"""
    if cv_preds is None:
        return
        
    print(f"\nCreating prediction visualization...")
    
    # Take first prediction window
    first_window = cv_preds[cv_preds['cutoff'] == cv_preds['cutoff'].iloc[0]]
    
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Time series
    plt.subplot(2, 2, 1)
    plt.plot(first_window['ds'], first_window['y'] - 273.15, 
             'o-', label='Actual', linewidth=2, markersize=6)
    if 'NHITS' in first_window.columns:
        plt.plot(first_window['ds'], first_window['NHITS'] - 273.15, 
                 's-', label='NHITS', linewidth=2, markersize=6)
    plt.title('Temperature Predictions (First Window)')
    plt.xlabel('Date')
    plt.ylabel('Temperature (°C)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # Plot 2: Scatter plot
    plt.subplot(2, 2, 2)
    if 'NHITS' in cv_preds.columns:
        plt.scatter(cv_preds['y'] - 273.15, cv_preds['NHITS'] - 273.15, alpha=0.6)
        min_temp = min(cv_preds['y'].min(), cv_preds['NHITS'].min()) - 273.15
        max_temp = max(cv_preds['y'].max(), cv_preds['NHITS'].max()) - 273.15
        plt.plot([min_temp, max_temp], [min_temp, max_temp], 'r--', alpha=0.8)
        plt.xlabel('Actual Temperature (°C)')
        plt.ylabel('Predicted Temperature (°C)')
        plt.title('Actual vs Predicted')
        plt.grid(True, alpha=0.3)
    
    # Plot 3: Residuals
    plt.subplot(2, 2, 3)
    if 'NHITS' in cv_preds.columns:
        residuals = cv_preds['NHITS'] - cv_preds['y']
        plt.scatter(cv_preds['y'] - 273.15, residuals, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Actual Temperature (°C)')
        plt.ylabel('Residuals (K)')
        plt.title('Residual Analysis')
        plt.grid(True, alpha=0.3)
    
    # Plot 4: Distribution comparison
    plt.subplot(2, 2, 4)
    plt.hist(cv_preds['y'] - 273.15, bins=20, alpha=0.5, label='Actual', density=True)
    if 'NHITS' in cv_preds.columns:
        plt.hist(cv_preds['NHITS'] - 273.15, bins=20, alpha=0.5, label='NHITS', density=True)
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Density')
    plt.title('Temperature Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_test_results.png', dpi=150, bbox_inches='tight')
    print("✓ Test results saved as 'model_test_results.png'")
    plt.show()

def main():
    """Main function"""
    print("Starting model debugging...")
    
    # Test minimal model
    cv_preds = test_minimal_model()
    
    # Create visualizations
    create_simple_prediction_plot(cv_preds)
    
    print("\n" + "="*60)
    print("MODEL TEST COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()