#!/usr/bin/env python3
"""
Simplified test script using only TimeXer model to debug zero prediction issue
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

def load_and_examine_data():
    """Load data and examine in detail"""
    print("="*60)
    print("DETAILED DATA EXAMINATION")
    print("="*60)
    
    # Load raw data
    geo_df = pd.read_csv("data/geo.csv")
    print(f"Raw data shape: {geo_df.shape}")
    
    # Check temperature column specifically
    temp_data = geo_df['temperature_2m']
    print(f"\nTemperature column analysis:")
    print(f"- Data type: {temp_data.dtype}")
    print(f"- Min: {temp_data.min():.6f}")
    print(f"- Max: {temp_data.max():.6f}")
    print(f"- Mean: {temp_data.mean():.6f}")
    print(f"- Std: {temp_data.std():.6f}")
    print(f"- First 10 values: {temp_data.head(10).tolist()}")
    print(f"- Any NaN: {temp_data.isna().sum()}")
    print(f"- Any zero: {(temp_data == 0).sum()}")
    print(f"- Any inf: {np.isinf(temp_data).sum()}")
    
    return geo_df

def preprocess_data_step_by_step(geo_df):
    """Preprocess data step by step with verification"""
    print("\n" + "="*60)
    print("STEP-BY-STEP PREPROCESSING")
    print("="*60)
    
    # Step 1: Time conversion
    print("Step 1: Converting time column...")
    geo_df['time'] = geo_df['time'].astype(str)
    geo_df['year'] = geo_df['time'].str[:4].astype(int)
    geo_df['month'] = geo_df['time'].str[4:].astype(int)
    geo_df['ds'] = pd.to_datetime(geo_df[['year', 'month']].assign(day=1))
    print(f"Date range: {geo_df['ds'].min()} to {geo_df['ds'].max()}")
    
    # Step 2: Add unique_id
    print("\nStep 2: Adding unique_id...")
    geo_df['unique_id'] = 'GEO'
    
    # Step 3: Rename target variable
    print("\nStep 3: Setting target variable...")
    original_temp = geo_df['temperature_2m'].copy()
    geo_df = geo_df.rename(columns={'temperature_2m': 'y'})
    print(f"Target variable (y) after rename:")
    print(f"- Min: {geo_df['y'].min():.6f}")
    print(f"- Max: {geo_df['y'].max():.6f}")
    print(f"- Mean: {geo_df['y'].mean():.6f}")
    print(f"- Same as original: {np.array_equal(original_temp.values, geo_df['y'].values)}")
    
    # Step 4: Select features
    print("\nStep 4: Selecting features...")
    feature_columns = [
        'surface_pressure',
        'u_component_of_wind_10m',
        'v_component_of_wind_10m',
        'surface_net_solar_radiation_sum',
        'total_precipitation_sum',
        'total_evaporation_sum'
    ]
    
    # Check each feature
    for feature in feature_columns:
        if feature in geo_df.columns:
            feat_data = geo_df[feature]
            print(f"  ✓ {feature}: range [{feat_data.min():.6f}, {feat_data.max():.6f}]")
        else:
            print(f"  ❌ {feature}: NOT FOUND")
    
    # Step 5: Create final dataset
    print("\nStep 5: Creating final dataset...")
    required_cols = ['unique_id', 'ds', 'y'] + feature_columns
    geo_final = geo_df[required_cols].copy()
    
    print(f"Before dropna: {geo_final.shape}")
    geo_final = geo_final.dropna()
    print(f"After dropna: {geo_final.shape}")
    
    # Final verification
    print(f"\nFinal target variable verification:")
    print(f"- Min: {geo_final['y'].min():.6f}")
    print(f"- Max: {geo_final['y'].max():.6f}")
    print(f"- Mean: {geo_final['y'].mean():.6f}")
    print(f"- Std: {geo_final['y'].std():.6f}")
    print(f"- First 5: {geo_final['y'].head().tolist()}")
    print(f"- Last 5: {geo_final['y'].tail().tolist()}")
    
    return geo_final, feature_columns

def test_timexer_minimal():
    """Test TimeXer with minimal configuration"""
    print("\n" + "="*60)
    print("TESTING TIMEXER MODEL")
    print("="*60)
    
    try:
        from neuralforecast.core import NeuralForecast
        from neuralforecast.models import TimeXer
        
        # Load and preprocess data
        geo_df = load_and_examine_data()
        geo_final, feature_columns = preprocess_data_step_by_step(geo_df)
        
        # Very conservative parameters
        HORIZON = 3  # Very short horizon
        INPUT_SIZE = 12  # Short input size
        
        print(f"\nModel configuration:")
        print(f"- Horizon: {HORIZON} months")
        print(f"- Input size: {INPUT_SIZE} months")
        print(f"- Features: {len(feature_columns)}")
        print(f"- Data points: {len(geo_final)}")
        
        # Check if we have enough data
        min_required = INPUT_SIZE + HORIZON * 2  # For 2 CV windows
        print(f"- Min required: {min_required}")
        print(f"- Sufficient: {'✓' if len(geo_final) >= min_required else '❌'}")
        
        if len(geo_final) < min_required:
            print("❌ Not enough data points!")
            return None
        
        # Create TimeXer model
        print(f"\nCreating TimeXer model...")
        model = TimeXer(
            h=HORIZON,
            input_size=INPUT_SIZE,
            n_series=1,
            futr_exog_list=feature_columns,
            patch_len=HORIZON,
            max_steps=50  # Very few steps for testing
        )
        
        print(f"✓ Model created successfully")
        
        # Set up NeuralForecast
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        nf = NeuralForecast(models=[model], freq="MS")
        
        print(f"\nStarting cross-validation (2 windows)...")
        
        # Run cross-validation
        cv_preds = nf.cross_validation(
            geo_final,
            step_size=HORIZON,
            n_windows=2  # Only 2 windows for testing
        )
        
        print(f"✅ Cross-validation completed!")
        print(f"Predictions shape: {cv_preds.shape}")
        print(f"Columns: {list(cv_preds.columns)}")
        
        # Detailed analysis of results
        print(f"\n" + "="*60)
        print("PREDICTION ANALYSIS")
        print("="*60)
        
        # Check actual values in predictions
        y_actual = cv_preds['y']
        print(f"Actual values in CV results:")
        print(f"- Shape: {y_actual.shape}")
        print(f"- Data type: {y_actual.dtype}")
        print(f"- Min: {y_actual.min():.6f}")
        print(f"- Max: {y_actual.max():.6f}")
        print(f"- Mean: {y_actual.mean():.6f}")
        print(f"- Std: {y_actual.std():.6f}")
        print(f"- First 5: {y_actual.head().tolist()}")
        print(f"- Any NaN: {y_actual.isna().sum()}")
        print(f"- Any zero: {(y_actual == 0).sum()}")
        print(f"- Any inf: {np.isinf(y_actual).sum()}")
        
        # Check TimeXer predictions
        if 'TimeXer' in cv_preds.columns:
            timexer_pred = cv_preds['TimeXer']
            print(f"\nTimeXer predictions:")
            print(f"- Shape: {timexer_pred.shape}")
            print(f"- Data type: {timexer_pred.dtype}")
            print(f"- Min: {timexer_pred.min():.6f}")
            print(f"- Max: {timexer_pred.max():.6f}")
            print(f"- Mean: {timexer_pred.mean():.6f}")
            print(f"- Std: {timexer_pred.std():.6f}")
            print(f"- First 5: {timexer_pred.head().tolist()}")
            print(f"- Any NaN: {timexer_pred.isna().sum()}")
            print(f"- Any zero: {(timexer_pred == 0).sum()}")
            print(f"- Any inf: {np.isinf(timexer_pred).sum()}")
            
            # Calculate simple metrics
            mae = np.mean(np.abs(y_actual - timexer_pred))
            rmse = np.sqrt(np.mean((y_actual - timexer_pred)**2))
            print(f"\nSimple metrics:")
            print(f"- MAE: {mae:.6f}")
            print(f"- RMSE: {rmse:.6f}")
            
        # Check dates and cutoffs
        print(f"\nDate information:")
        print(f"- Date range in CV: {cv_preds['ds'].min()} to {cv_preds['ds'].max()}")
        print(f"- Unique cutoffs: {cv_preds['cutoff'].nunique()}")
        print(f"- Cutoff values: {cv_preds['cutoff'].unique()}")
        
        return cv_preds
        
    except Exception as e:
        print(f"❌ Error in TimeXer testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def create_debug_plots(cv_preds, geo_final):
    """Create debug plots"""
    print(f"\n" + "="*60)
    print("CREATING DEBUG PLOTS")
    print("="*60)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('TimeXer Debug Analysis', fontsize=16)
    
    # Plot 1: Original data
    ax1 = axes[0, 0]
    temp_celsius = geo_final['y'] - 273.15
    ax1.plot(geo_final['ds'], temp_celsius, alpha=0.7, linewidth=1)
    ax1.set_title('Original Temperature Data')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Temperature (°C)')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: Original data distribution
    ax2 = axes[0, 1]
    ax2.hist(temp_celsius, bins=30, alpha=0.7, color='blue', edgecolor='black')
    ax2.set_title('Original Temperature Distribution')
    ax2.set_xlabel('Temperature (°C)')
    ax2.set_ylabel('Frequency')
    ax2.grid(True, alpha=0.3)
    
    if cv_preds is not None and len(cv_preds) > 0:
        # Plot 3: CV actual values
        ax3 = axes[0, 2]
        cv_temp_celsius = cv_preds['y'] - 273.15
        ax3.plot(cv_preds['ds'], cv_temp_celsius, 'o-', alpha=0.7)
        ax3.set_title('CV Actual Values')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Temperature (°C)')
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(axis='x', rotation=45)
        
        # Plot 4: CV actual distribution
        ax4 = axes[1, 0]
        ax4.hist(cv_temp_celsius, bins=20, alpha=0.7, color='green', edgecolor='black')
        ax4.set_title('CV Actual Distribution')
        ax4.set_xlabel('Temperature (°C)')
        ax4.set_ylabel('Frequency')
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Predictions vs actual (if available)
        ax5 = axes[1, 1]
        if 'TimeXer' in cv_preds.columns:
            pred_celsius = cv_preds['TimeXer'] - 273.15
            ax5.scatter(cv_temp_celsius, pred_celsius, alpha=0.6)
            min_temp = min(cv_temp_celsius.min(), pred_celsius.min())
            max_temp = max(cv_temp_celsius.max(), pred_celsius.max())
            ax5.plot([min_temp, max_temp], [min_temp, max_temp], 'r--')
            ax5.set_xlabel('Actual Temperature (°C)')
            ax5.set_ylabel('Predicted Temperature (°C)')
            ax5.set_title('Actual vs Predicted')
            ax5.grid(True, alpha=0.3)
        else:
            ax5.text(0.5, 0.5, 'No TimeXer predictions', ha='center', va='center', transform=ax5.transAxes)
            ax5.set_title('No Predictions Available')
        
        # Plot 6: Time series comparison
        ax6 = axes[1, 2]
        first_window = cv_preds[cv_preds['cutoff'] == cv_preds['cutoff'].iloc[0]]
        ax6.plot(first_window['ds'], first_window['y'] - 273.15, 'o-', label='Actual', linewidth=2)
        if 'TimeXer' in cv_preds.columns:
            ax6.plot(first_window['ds'], first_window['TimeXer'] - 273.15, 's-', label='TimeXer', linewidth=2)
        ax6.set_title('First Prediction Window')
        ax6.set_xlabel('Date')
        ax6.set_ylabel('Temperature (°C)')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        ax6.tick_params(axis='x', rotation=45)
    else:
        # Fill remaining plots with "No data" message
        for ax in [axes[0, 2], axes[1, 0], axes[1, 1], axes[1, 2]]:
            ax.text(0.5, 0.5, 'No CV data available', ha='center', va='center', transform=ax.transAxes)
    
    plt.tight_layout()
    plt.savefig('timexer_debug.png', dpi=150, bbox_inches='tight')
    print("✓ Debug plots saved as 'timexer_debug.png'")
    plt.show()

def main():
    """Main function"""
    print("Starting TimeXer-only debugging...")
    
    # Test TimeXer model
    cv_preds = test_timexer_minimal()
    
    # Load original data for comparison
    geo_df = pd.read_csv("data/geo.csv")
    geo_df['ds'] = pd.to_datetime(pd.DataFrame({
        'year': geo_df['time'].astype(str).str[:4].astype(int),
        'month': geo_df['time'].astype(str).str[4:].astype(int),
        'day': 1
    }))
    geo_df['y'] = geo_df['temperature_2m']
    
    # Create debug plots
    create_debug_plots(cv_preds, geo_df)
    
    print("\n" + "="*60)
    print("DEBUG SESSION COMPLETE")
    print("="*60)
    
    if cv_preds is not None:
        print("✅ TimeXer model ran successfully")
        print(f"✅ Generated {len(cv_preds)} predictions")
        
        # Summary of findings
        if (cv_preds['y'] == 0).all():
            print("❌ ALL ACTUAL VALUES ARE ZERO - This is the problem!")
        elif (cv_preds['y'] == 0).any():
            print(f"⚠ Some actual values are zero: {(cv_preds['y'] == 0).sum()} out of {len(cv_preds)}")
        else:
            print("✅ Actual values look normal")
            
        if 'TimeXer' in cv_preds.columns:
            if (cv_preds['TimeXer'] == 0).all():
                print("❌ ALL PREDICTIONS ARE ZERO")
            elif (cv_preds['TimeXer'] == 0).any():
                print(f"⚠ Some predictions are zero: {(cv_preds['TimeXer'] == 0).sum()} out of {len(cv_preds)}")
            else:
                print("✅ Predictions look normal")
    else:
        print("❌ TimeXer model failed to run")

if __name__ == "__main__":
    main()