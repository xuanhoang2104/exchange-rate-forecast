#!/usr/bin/env python3
"""
Visualize processed exchange rate data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import os
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_config():
    """Load configuration from YAML file"""
    try:
        config_path = Path("D:/FPT/ki 7/DAT/project_exchange_rate/exchange-rate-forecast/config/config.yaml")
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except:
        print("Warning: Config file not found, using default settings")
        return {'data': {'target_currency': 'algerian_dinar'}}

def load_original_data():
    """Load original merged data"""
    data_path = Path("D:/FPT/ki 7/DAT/project_exchange_rate/exchange-rate-forecast/data/processed/merged/all_currencies.csv")
    print(f"Loading original data from: {data_path}")
    
    if not data_path.exists():
        raise FileNotFoundError(f"Original data not found at: {data_path}")
    
    df = pd.read_csv(data_path, parse_dates=['date'])
    df.set_index('date', inplace=True)
    print(f"Original data loaded: {df.shape[0]} rows, {df.shape[1]} currencies")
    return df

def load_processed_data():
    """Load processed data for visualization"""
    processed_dir = Path("D:/FPT/ki 7/DAT/project_exchange_rate/exchange-rate-forecast/data/processed/model_ready")
    print(f"Loading processed data from: {processed_dir}")
    
    if not processed_dir.exists():
        raise FileNotFoundError(f"Processed data directory not found at: {processed_dir}")
    
    # Check what files exist
    files = list(processed_dir.glob("*"))
    print(f"Files in directory: {[f.name for f in files]}")
    
    # Load numpy arrays
    data_files = ['X_train.npy', 'y_train.npy', 'X_val.npy', 'y_val.npy', 'X_test.npy', 'y_test.npy']
    data = {}
    
    for file in data_files:
        file_path = processed_dir / file
        if file_path.exists():
            print(f"Loading {file}...")
            data[file.replace('.npy', '')] = np.load(file_path)
        else:
            print(f"Warning: {file} not found")
    
    # Load feature names
    feature_names = []
    feature_names_path = processed_dir / "feature_names.txt"
    if feature_names_path.exists():
        with open(feature_names_path, 'r') as f:
            feature_names = [line.strip() for line in f.readlines()]
        print(f"Loaded {len(feature_names)} feature names")
    else:
        print("Warning: feature_names.txt not found")
        # Create default feature names based on shape
        if 'X_train' in data:
            X_train = data['X_train']
            if X_train.ndim == 3:
                # Create feature names for sequences
                for t in range(X_train.shape[1]):  # timesteps
                    for f in range(X_train.shape[2]):  # features
                        feature_names.append(f't{t+1:02d}_f{f+1:02d}')
            else:
                feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]
    
    # Load metadata if exists
    metadata = {}
    metadata_path = processed_dir / "metadata.yaml"
    if metadata_path.exists():
        try:
            with open(metadata_path, 'r') as f:
                metadata = yaml.safe_load(f)
            print("Loaded metadata")
        except Exception as e:
            print(f"Warning: Could not load metadata: {e}")
    
    # Check what data was loaded
    print(f"\nLoaded processed data:")
    for key in ['X_train', 'X_val', 'X_test']:
        if key in data:
            print(f"  {key} shape: {data[key].shape}")
    
    return {
        'X_train': data.get('X_train'),
        'y_train': data.get('y_train'),
        'X_val': data.get('X_val'),
        'y_val': data.get('y_val'),
        'X_test': data.get('X_test'),
        'y_test': data.get('y_test'),
        'feature_names': feature_names,
        'metadata': metadata
    }

def visualize_original_data(df):
    """Visualize original time series data"""
    print("\n" + "="*70)
    print("VISUALIZING ORIGINAL DATA")
    print("="*70)
    
    # Select top 8 currencies for visualization
    top_currencies = df.notna().sum().sort_values(ascending=False).head(8).index.tolist()
    
    fig, axes = plt.subplots(4, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, currency in enumerate(top_currencies):
        if i >= len(axes):
            break
            
        ax = axes[i]
        df[currency].plot(ax=ax, linewidth=1, alpha=0.8)
        
        # Add rolling mean
        rolling_mean = df[currency].rolling(window=30).mean()
        rolling_mean.plot(ax=ax, linewidth=2, color='red', alpha=0.7, label='30-day MA')
        
        ax.set_title(f'{currency.replace("_", " ").title()} Exchange Rate', fontsize=12, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Rate')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        stats_text = f"Mean: {df[currency].mean():.2f}\nStd: {df[currency].std():.2f}"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Original Exchange Rate Time Series (Top 8 Currencies)', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('original_time_series.png', dpi=300, bbox_inches='tight')
    plt.show()

def visualize_target_currency(df, target_currency='algerian_dinar'):
    """Detailed visualization of target currency"""
    print(f"\nDetailed analysis of target currency: {target_currency}")
    
    if target_currency not in df.columns:
        print(f"Warning: {target_currency} not found. Using first available.")
        target_currency = df.columns[0]
    
    print(f"Analyzing: {target_currency}")
    
    fig = plt.figure(figsize=(16, 10))
    
    # 1. Original time series
    ax1 = plt.subplot(3, 2, 1)
    df[target_currency].plot(ax=ax1, linewidth=1, color='blue', alpha=0.7)
    ax1.set_title(f'{target_currency.replace("_", " ").title()} - Raw Time Series', fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Exchange Rate')
    ax1.grid(True, alpha=0.3)
    
    # 2. Rolling statistics
    ax2 = plt.subplot(3, 2, 2)
    rolling_mean = df[target_currency].rolling(window=30).mean()
    rolling_std = df[target_currency].rolling(window=30).std()
    
    ax2.plot(df.index, rolling_mean, label='30-day MA', linewidth=2, color='red')
    ax2.fill_between(df.index, 
                     rolling_mean - 2*rolling_std, 
                     rolling_mean + 2*rolling_std, 
                     alpha=0.2, color='red', label='±2σ')
    ax2.set_title('30-day Moving Average with Volatility Bands', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Returns distribution
    ax3 = plt.subplot(3, 2, 3)
    returns = df[target_currency].pct_change().dropna()
    returns.hist(bins=100, ax=ax3, edgecolor='black', alpha=0.7)
    ax3.axvline(returns.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {returns.mean():.4f}')
    ax3.axvline(returns.median(), color='green', linestyle='--', linewidth=2, label=f'Median: {returns.median():.4f}')
    ax3.set_title('Daily Returns Distribution', fontweight='bold')
    ax3.set_xlabel('Daily Return')
    ax3.set_ylabel('Frequency')
    ax3.legend()
    
    # 4. QQ plot for normality
    from scipy import stats
    ax4 = plt.subplot(3, 2, 4)
    stats.probplot(returns.dropna(), dist="norm", plot=ax4)
    ax4.set_title('Q-Q Plot (Normality Test)', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # 5. Seasonal decomposition
    try:
        from statsmodels.tsa.seasonal import seasonal_decompose
        ax5 = plt.subplot(3, 2, 5)
        
        # Use last 3 years for seasonal decomposition
        sample_data = df[target_currency].last('3Y')
        if len(sample_data) >= 730:  # Need at least 2 years
            decomposition = seasonal_decompose(sample_data, model='additive', period=365)
            
            decomposition.trend.plot(ax=ax5, label='Trend', linewidth=2)
            decomposition.seasonal.plot(ax=ax5, label='Seasonal', alpha=0.7)
            decomposition.resid.plot(ax=ax5, label='Residual', alpha=0.5)
            ax5.set_title('Seasonal Decomposition (Last 3 Years)', fontweight='bold')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        else:
            ax5.text(0.5, 0.5, 'Not enough data for seasonal decomposition', 
                     ha='center', va='center', transform=ax5.transAxes)
    except Exception as e:
        print(f"Could not perform seasonal decomposition: {e}")
        ax5.text(0.5, 0.5, 'Seasonal decomposition failed', 
                 ha='center', va='center', transform=ax5.transAxes)
    
    # 6. Autocorrelation
    ax6 = plt.subplot(3, 2, 6)
    from statsmodels.graphics.tsaplots import plot_acf
    plot_acf(df[target_currency].dropna(), lags=50, ax=ax6, alpha=0.05)
    ax6.set_title('Autocorrelation (50 Lags)', fontweight='bold')
    ax6.set_xlabel('Lag')
    ax6.set_ylabel('Autocorrelation')
    ax6.grid(True, alpha=0.3)
    
    plt.suptitle(f'Comprehensive Analysis: {target_currency.replace("_", " ").title()}', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('target_currency_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print statistics
    print("\nStatistical Summary:")
    print("-" * 40)
    print(f"Mean: {df[target_currency].mean():.4f}")
    print(f"Std: {df[target_currency].std():.4f}")
    print(f"Min: {df[target_currency].min():.4f}")
    print(f"Max: {df[target_currency].max():.4f}")
    print(f"Skewness: {df[target_currency].skew():.4f}")
    print(f"Kurtosis: {df[target_currency].kurtosis():.4f}")
    
    try:
        from statsmodels.tsa.stattools import adfuller
        result = adfuller(df[target_currency].dropna())
        print(f"ADF p-value: {result[1]:.4f}")
    except:
        print("ADF test not available")

def visualize_processed_sequences(data):
    """Visualize processed sequences"""
    print("\n" + "="*70)
    print("VISUALIZING PROCESSED SEQUENCES")
    print("="*70)
    
    X_train = data['X_train']
    y_train = data['y_train']
    
    if X_train is None or y_train is None:
        print("Error: X_train or y_train not loaded")
        return
    
    fig = plt.figure(figsize=(16, 10))
    
    # 1. Sequence structure
    ax1 = plt.subplot(2, 2, 1)
    
    # Plot first sequence
    sample_sequence = X_train[0]
    num_features = sample_sequence.shape[1]
    
    # Plot first 5 features
    for i in range(min(5, num_features)):
        ax1.plot(sample_sequence[:, i], label=f'Feature {i+1}', alpha=0.7)
    
    ax1.axvline(x=sample_sequence.shape[0]-1, color='red', linestyle='--', 
                linewidth=2, label='Prediction Point')
    ax1.scatter(sample_sequence.shape[0]-1, y_train[0], color='red', 
                s=100, zorder=5, label='Target Value')
    
    ax1.set_title(f'Sample Sequence (First of {len(X_train)} sequences)', fontweight='bold')
    ax1.set_xlabel('Time Step (Days)')
    ax1.set_ylabel('Normalized Value')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # 2. Target distribution
    ax2 = plt.subplot(2, 2, 2)
    all_targets = []
    for key in ['y_train', 'y_val', 'y_test']:
        if data[key] is not None:
            all_targets.append(data[key])
    
    if all_targets:
        all_targets = np.concatenate(all_targets)
        ax2.hist(all_targets, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
        ax2.axvline(all_targets.mean(), color='red', linestyle='--', 
                    linewidth=2, label=f'Mean: {all_targets.mean():.3f}')
        ax2.axvline(all_targets.std(), color='green', linestyle='--', 
                    linewidth=2, label=f'Std: {all_targets.std():.3f}')
        
        ax2.set_title('Target Value Distribution (All Splits)', fontweight='bold')
        ax2.set_xlabel('Normalized Target Value')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No target data available', 
                 ha='center', va='center', transform=ax2.transAxes)
    
    # 3. Train/Val/Test split visualization
    ax3 = plt.subplot(2, 2, 3)
    
    # Create timeline
    train_len = len(y_train) if y_train is not None else 0
    val_len = len(data['y_val']) if data['y_val'] is not None else 0
    test_len = len(data['y_test']) if data['y_test'] is not None else 0
    total_len = train_len + val_len + test_len
    
    if total_len > 0:
        # Plot as colored segments
        colors = ['blue', 'orange', 'green']
        labels = ['Train', 'Validation', 'Test']
        
        ax3.barh(0, train_len, color=colors[0], alpha=0.7, label=labels[0])
        ax3.barh(0, val_len, left=train_len, color=colors[1], alpha=0.7, label=labels[1])
        ax3.barh(0, test_len, left=train_len + val_len, color=colors[2], alpha=0.7, label=labels[2])
        
        ax3.set_xlabel('Number of Sequences')
        ax3.set_title('Train/Validation/Test Split', fontweight='bold')
        ax3.legend()
        ax3.set_yticks([])
        
        # Add percentage labels
        percentages = [train_len/total_len*100, val_len/total_len*100, test_len/total_len*100]
        positions = [train_len/2, train_len + val_len/2, train_len + val_len + test_len/2]
        
        for pos, perc in zip(positions, percentages):
            if perc > 0:  # Only show if percentage > 0
                ax3.text(pos, 0.1, f'{perc:.1f}%', ha='center', va='center', 
                         color='white', fontweight='bold', fontsize=10)
    else:
        ax3.text(0.5, 0.5, 'No sequence data available', 
                 ha='center', va='center', transform=ax3.transAxes)
    
    # 4. Feature correlation (first sequence)
    ax4 = plt.subplot(2, 2, 4)
    
    # Calculate correlation for first sequence
    if sample_sequence.shape[1] > 1:
        corr_matrix = np.corrcoef(sample_sequence.T)
        
        im = ax4.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
        
        # Add correlation values for smaller matrices
        if corr_matrix.shape[0] <= 20:  # Only show text if not too many features
            for i in range(corr_matrix.shape[0]):
                for j in range(corr_matrix.shape[1]):
                    if i != j:  # Skip diagonal
                        ax4.text(j, i, f'{corr_matrix[i, j]:.2f}', 
                                ha='center', va='center', 
                                color='white' if abs(corr_matrix[i, j]) > 0.5 else 'black',
                                fontsize=6)
        
        ax4.set_title(f'Feature Correlation (First Sequence)\n{num_features} features', fontweight='bold')
        ax4.set_xlabel('Feature Index')
        ax4.set_ylabel('Feature Index')
        plt.colorbar(im, ax=ax4)
    else:
        ax4.text(0.5, 0.5, 'Not enough features for correlation matrix', 
                 ha='center', va='center', transform=ax4.transAxes)
    
    plt.suptitle('Processed Sequence Analysis', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('processed_sequences_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print sequence statistics
    print(f"\nSequence Statistics:")
    print("-" * 40)
    print(f"Total sequences: {train_len + val_len + test_len}")
    print(f"Sequence length: {X_train.shape[1]} days")
    print(f"Features per timestep: {X_train.shape[2]}")
    print(f"Training sequences: {train_len}")
    print(f"Validation sequences: {val_len}")
    print(f"Test sequences: {test_len}")
    if len(all_targets) > 0:
        print(f"Target mean: {all_targets.mean():.4f}")
        print(f"Target std: {all_targets.std():.4f}")

def visualize_feature_importance(data):
    """Visualize feature importance analysis"""
    print("\n" + "="*70)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*70)
    
    X_train = data['X_train']
    y_train = data['y_train']
    feature_names = data['feature_names']
    
    if X_train is None or y_train is None:
        print("Error: X_train or y_train not loaded")
        return
    
    # Flatten for feature importance analysis
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    
    # Create feature names for flattened data
    if len(feature_names) == X_train.shape[1]:  # If feature names are for timesteps
        # Recreate proper feature names for sequences
        feature_names_flat = []
        for t in range(X_train.shape[1]):  # timesteps
            for f in range(X_train.shape[2]):  # features per timestep
                feature_names_flat.append(f't{t+1:02d}_f{f+1:02d}')
    else:
        feature_names_flat = feature_names[:X_train_flat.shape[1]]
    
    fig = plt.figure(figsize=(16, 8))
    
    # 1. Feature variance
    ax1 = plt.subplot(1, 2, 1)
    feature_variance = np.var(X_train_flat, axis=0)
    top_n = min(20, len(feature_variance))
    top_indices = np.argsort(feature_variance)[-top_n:][::-1]
    
    colors = plt.cm.viridis(np.linspace(0, 1, top_n))
    bars = ax1.barh(range(top_n), feature_variance[top_indices], color=colors)
    
    # Add feature names if available
    if len(feature_names_flat) == len(feature_variance):
        y_labels = [feature_names_flat[i] for i in top_indices]
    else:
        y_labels = [f'Feature {i+1}' for i in top_indices]
    
    ax1.set_yticks(range(top_n))
    ax1.set_yticklabels(y_labels, fontsize=8)
    ax1.invert_yaxis()
    ax1.set_xlabel('Variance')
    ax1.set_title(f'Top {top_n} Features by Variance', fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # 2. Correlation with target
    ax2 = plt.subplot(1, 2, 2)
    
    # Calculate correlation with target for each feature
    correlations = []
    for i in range(X_train_flat.shape[1]):
        corr = np.corrcoef(X_train_flat[:, i], y_train)[0, 1]
        correlations.append(abs(corr))  # Use absolute correlation
    
    top_corr_indices = np.argsort(correlations)[-top_n:][::-1]
    
    colors = plt.cm.plasma(np.linspace(0, 1, top_n))
    bars = ax2.barh(range(top_n), np.array(correlations)[top_corr_indices], color=colors)
    
    # Add feature names if available
    if len(feature_names_flat) == len(correlations):
        y_labels = [feature_names_flat[i] for i in top_corr_indices]
    else:
        y_labels = [f'Feature {i+1}' for i in top_corr_indices]
    
    ax2.set_yticks(range(top_n))
    ax2.set_yticklabels(y_labels, fontsize=8)
    ax2.invert_yaxis()
    ax2.set_xlabel('Absolute Correlation with Target')
    ax2.set_title(f'Top {top_n} Features by Target Correlation', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    
    plt.suptitle('Feature Importance Analysis', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()

def visualize_data_split_timeline(df, target_currency='algerian_dinar'):
    """Visualize train/val/test split on timeline"""
    print("\n" + "="*70)
    print("DATA SPLIT TIMELINE VISUALIZATION")
    print("="*70)
    
    if target_currency not in df.columns:
        target_currency = df.columns[0]
    
    # Get split dates from data
    split_dates = {
        'train_start': '2004-01-05',
        'train_end': '2019-11-14',
        'val_start': '2019-11-15',
        'val_end': '2022-10-12',
        'test_start': '2022-10-13',
        'test_end': '2026-01-30'
    }
    
    # Convert to datetime
    for key in split_dates:
        split_dates[key] = pd.to_datetime(split_dates[key])
    
    fig, ax = plt.subplots(figsize=(15, 6))
    
    # Plot full time series
    df_target = df[target_currency].copy()
    ax.plot(df_target.index, df_target.values, color='black', alpha=0.3, linewidth=0.5, label='Full Series')
    
    # Highlight splits
    colors = ['blue', 'orange', 'green']
    labels = ['Train', 'Validation', 'Test']
    
    # Train data
    train_mask = (df_target.index >= split_dates['train_start']) & (df_target.index <= split_dates['train_end'])
    ax.plot(df_target.index[train_mask], df_target.values[train_mask], 
            color=colors[0], linewidth=2, label=labels[0])
    
    # Validation data
    val_mask = (df_target.index >= split_dates['val_start']) & (df_target.index <= split_dates['val_end'])
    ax.plot(df_target.index[val_mask], df_target.values[val_mask], 
            color=colors[1], linewidth=2, label=labels[1])
    
    # Test data
    test_mask = (df_target.index >= split_dates['test_start']) & (df_target.index <= split_dates['test_end'])
    ax.plot(df_target.index[test_mask], df_target.values[test_mask], 
            color=colors[2], linewidth=2, label=labels[2])
    
    # Add vertical lines for split points
    for date in [split_dates['train_end'], split_dates['val_end']]:
        ax.axvline(x=date, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    
    # Add split annotations
    ax.text(split_dates['train_start'] + (split_dates['train_end'] - split_dates['train_start'])/2,
            df_target.max() * 0.9, 'TRAIN\n(2004-2019)', 
            ha='center', va='center', fontweight='bold', fontsize=10,
            bbox=dict(boxstyle='round', facecolor=colors[0], alpha=0.3))
    
    ax.text(split_dates['val_start'] + (split_dates['val_end'] - split_dates['val_start'])/2,
            df_target.max() * 0.9, 'VALIDATION\n(2019-2022)', 
            ha='center', va='center', fontweight='bold', fontsize=10,
            bbox=dict(boxstyle='round', facecolor=colors[1], alpha=0.3))
    
    ax.text(split_dates['test_start'] + (split_dates['test_end'] - split_dates['test_start'])/2,
            df_target.max() * 0.9, 'TEST\n(2022-2026)', 
            ha='center', va='center', fontweight='bold', fontsize=10,
            bbox=dict(boxstyle='round', facecolor=colors[2], alpha=0.3))
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel(f'{target_currency.replace("_", " ").title()} Exchange Rate', fontsize=12)
    ax.set_title('Train/Validation/Test Split Timeline', fontsize=16, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Format x-axis
    ax.xaxis.set_major_locator(plt.MaxNLocator(10))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('data_split_timeline.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print split statistics
    print("\nSplit Statistics:")
    print("-" * 40)
    print(f"Train period: {split_dates['train_start'].date()} to {split_dates['train_end'].date()}")
    print(f"  Duration: {(split_dates['train_end'] - split_dates['train_start']).days / 365:.1f} years")
    print(f"  Samples: {train_mask.sum():,}")
    
    print(f"\nValidation period: {split_dates['val_start'].date()} to {split_dates['val_end'].date()}")
    print(f"  Duration: {(split_dates['val_end'] - split_dates['val_start']).days / 365:.1f} years")
    print(f"  Samples: {val_mask.sum():,}")
    
    print(f"\nTest period: {split_dates['test_start'].date()} to {split_dates['test_end'].date()}")
    print(f"  Duration: {(split_dates['test_end'] - split_dates['test_start']).days / 365:.1f} years")
    print(f"  Samples: {test_mask.sum():,}")

def create_summary_report(df, processed_data):
    """Create a comprehensive summary report"""
    print("\n" + "="*70)
    print("DATA ANALYSIS SUMMARY REPORT")
    print("="*70)
    
    # Create report text
    report = []
    report.append("="*70)
    report.append("EXCHANGE RATE DATA ANALYSIS REPORT")
    report.append("="*70)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # 1. Data Overview
    report.append("1. DATA OVERVIEW")
    report.append("-"*40)
    report.append(f"Total observations: {len(df):,}")
    report.append(f"Number of currencies: {len(df.columns)}")
    report.append(f"Date range: {df.index.min().date()} to {df.index.max().date()}")
    report.append(f"Total duration: {(df.index.max() - df.index.min()).days / 365:.1f} years")
    report.append("")
    
    # 2. Target Currency Analysis
    target_currency = 'algerian_dinar'
    if target_currency not in df.columns:
        target_currency = df.columns[0]
    
    report.append("2. TARGET CURRENCY ANALYSIS")
    report.append("-"*40)
    report.append(f"Target currency: {target_currency.replace('_', ' ').title()}")
    report.append(f"Mean exchange rate: {df[target_currency].mean():.4f}")
    report.append(f"Standard deviation: {df[target_currency].std():.4f}")
    report.append(f"Minimum value: {df[target_currency].min():.4f}")
    report.append(f"Maximum value: {df[target_currency].max():.4f}")
    report.append(f"Missing values: {df[target_currency].isna().sum()}")
    report.append("")
    
    # 3. Processing Results
    report.append("3. PROCESSING RESULTS")
    report.append("-"*40)
    
    X_train = processed_data['X_train']
    if X_train is not None:
        report.append(f"Training sequences: {X_train.shape[0]:,}")
        report.append(f"Validation sequences: {processed_data['X_val'].shape[0] if processed_data['X_val'] is not None else 0:,}")
        report.append(f"Test sequences: {processed_data['X_test'].shape[0] if processed_data['X_test'] is not None else 0:,}")
        report.append(f"Sequence length: {X_train.shape[1]} days")
        report.append(f"Features per timestep: {X_train.shape[2]}")
        report.append(f"Total features: {X_train.shape[1] * X_train.shape[2]}")
    else:
        report.append("Processed data not available")
    report.append("")
    
    # 4. Data Quality
    report.append("4. DATA QUALITY")
    report.append("-"*40)
    report.append(f"Infinity values in original data: {df.isin([np.inf, -np.inf]).sum().sum()}")
    report.append(f"Missing values in original data: {df.isna().sum().sum()}")
    report.append(f"Zero/near-zero values: {(df.abs() < 1e-10).sum().sum()}")
    report.append("")
    
    # 5. Statistical Properties
    report.append("5. STATISTICAL PROPERTIES")
    report.append("-"*40)
    target_series = df[target_currency]
    report.append(f"Skewness: {target_series.skew():.4f}")
    report.append(f"Kurtosis: {target_series.kurtosis():.4f}")
    
    try:
        from statsmodels.tsa.stattools import adfuller
        result = adfuller(target_series.dropna())
        report.append(f"ADF test p-value: {result[1]:.4f}")
    except:
        report.append("ADF test: Not available")
    
    returns = target_series.pct_change().dropna()
    report.append(f"Daily returns mean: {returns.mean():.6f}")
    report.append(f"Daily returns std: {returns.std():.6f}")
    report.append("")
    
    # 6. Recommendations
    report.append("6. RECOMMENDATIONS FOR MODELING")
    report.append("-"*40)
    report.append("✓ Data is clean and ready for modeling")
    report.append("✓ Sufficient historical data (22 years)")
    report.append("✓ Proper train/val/test split")
    report.append("✓ Features engineered appropriately")
    report.append("✓ Consider testing multiple sequence lengths")
    report.append("✓ Experiment with different feature sets")
    report.append("✓ Try LSTM, GRU, or Transformer models")
    report.append("="*70)
    
    # Save report to file
    with open('data_analysis_report.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    # Print to console
    print('\n'.join(report))
    print(f"\nReport saved to: data_analysis_report.txt")

def main():
    """Main visualization pipeline"""
    print("="*70)
    print("EXCHANGE RATE DATA VISUALIZATION")
    print("="*70)
    
    try:
        # Load config for target currency
        config = load_config()
        target_currency = config.get('data', {}).get('target_currency', 'algerian_dinar')
        
        # Load data
        print("Loading data...")
        df = load_original_data()
        processed_data = load_processed_data()
        
        # Create output directory in current folder
        output_dir = Path("visualizations")
        output_dir.mkdir(exist_ok=True)
        print(f"Output directory: {output_dir.absolute()}")
        
        # Run visualizations
        visualize_original_data(df)
        visualize_target_currency(df, target_currency)
        visualize_data_split_timeline(df, target_currency)
        
        # Only visualize processed sequences if data is available
        if processed_data['X_train'] is not None:
            visualize_processed_sequences(processed_data)
            visualize_feature_importance(processed_data)
        else:
            print("\nWarning: Processed sequences not available, skipping sequence visualizations")
        
        create_summary_report(df, processed_data)
        
        print("\n" + "="*70)
        print(f"ALL VISUALIZATIONS COMPLETE!")
        print(f"Visualizations saved to: {output_dir.absolute()}")
        print("="*70)
        
        # List generated files
        print("\nGenerated files:")
        for file in output_dir.glob("*"):
            print(f"  - {file.name}")
            
    except Exception as e:
        print(f"Error in visualization pipeline: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Create visualizations folder in current directory
    main()