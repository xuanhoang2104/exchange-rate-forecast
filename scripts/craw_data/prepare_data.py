#!/usr/bin/env python3
"""
Data Preparation Script for Exchange Rate Forecasting Project

This script splits the raw exchange rate data (all currencies in one file)
into individual CSV files for each currency.

Usage:
    python scripts/prepare_data.py --input data/raw/exchange_rate_usd_to.csv
"""

import pandas as pd
import os
import argparse
from pathlib import Path
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def split_currency_data(input_file: str, output_dir: str = "data/processed/currencies"):
    """
    Split exchange rate data into individual CSV files for each currency.
    
    Args:
        input_file: Path to the raw CSV file containing all currencies
        output_dir: Directory to save individual currency files
    
    Returns:
        Dictionary with processing statistics
    """
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Reading data from: {input_file}")
    
    try:
        # Read the CSV file
        df = pd.read_csv(input_file)
        
        # The first column should be date
        date_column = df.columns[0]
        
        # Get all currency columns (all columns except the first one)
        currencies = df.columns[1:]
        
        logger.info(f"Found {len(currencies)} currency columns")
        logger.info(f"Total rows: {len(df)}")
        
        statistics = {
            'total_currencies': len(currencies),
            'processed_currencies': 0,
            'total_rows': len(df),
            'currency_files': []
        }
        
        # Process each currency
        for currency in currencies:
            # Create DataFrame for this currency
            currency_df = pd.DataFrame({
                'date': df[date_column],
                'rate': df[currency]
            })
            
            # Remove rows with NaN values
            currency_df = currency_df.dropna()
            
            if len(currency_df) > 0:
                # Create safe filename
                safe_name = currency.lower().replace(' ', '_').replace(',', '')
                output_file = output_path / f"{safe_name}.csv"
                
                # Save to CSV
                currency_df.to_csv(output_file, index=False)
                
                # Calculate basic statistics
                first_date = currency_df['date'].iloc[0]
                last_date = currency_df['date'].iloc[-1]
                first_rate = currency_df['rate'].iloc[0]
                last_rate = currency_df['rate'].iloc[-1]
                
                statistics['currency_files'].append({
                    'currency': currency,
                    'filename': f"{safe_name}.csv",
                    'rows': len(currency_df),
                    'first_date': first_date,
                    'last_date': last_date,
                    'first_rate': first_rate,
                    'last_rate': last_rate,
                    'mean_rate': currency_df['rate'].mean(),
                    'std_rate': currency_df['rate'].std()
                })
                
                statistics['processed_currencies'] += 1
                
                logger.debug(f"Saved {currency}: {len(currency_df)} rows ({first_date} to {last_date})")
            else:
                logger.warning(f"No data for currency: {currency}")
        
        # Create a summary file
        summary_df = pd.DataFrame(statistics['currency_files'])
        summary_file = output_path / "currency_summary.csv"
        summary_df.to_csv(summary_file, index=False)
        
        logger.info(f"Processed {statistics['processed_currencies']} currencies")
        logger.info(f"Saved files to: {output_path}")
        logger.info(f"Summary file: {summary_file}")
        
        return statistics
        
    except Exception as e:
        logger.error(f"Error processing file: {e}")
        raise

def create_combined_dataset(currency_dir: str, output_file: str = "data/processed/combined_currencies.csv"):
    """
    Create a combined dataset with all currencies in one file (long format).
    
    Args:
        currency_dir: Directory containing individual currency files
        output_file: Path for the combined output file
    """
    
    logger.info(f"Creating combined dataset from: {currency_dir}")
    
    all_data = []
    
    # Get all currency CSV files
    currency_files = list(Path(currency_dir).glob("*.csv"))
    currency_files = [f for f in currency_files if f.name != "currency_summary.csv"]
    
    for file_path in currency_files:
        # Extract currency name from filename
        currency_name = file_path.stem.replace('usd_to_', '') if 'usd_to_' in file_path.stem else file_path.stem
        
        # Read the currency data
        df = pd.read_csv(file_path)
        
        # Add currency column
        df['currency'] = currency_name
        
        all_data.append(df)
    
    if all_data:
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Reorder columns
        combined_df = combined_df[['date', 'currency', 'rate']]
        
        # Save combined dataset
        combined_df.to_csv(output_file, index=False)
        
        logger.info(f"Combined dataset saved: {output_file}")
        logger.info(f"Total records: {len(combined_df)}")
        logger.info(f"Unique currencies: {combined_df['currency'].nunique()}")
        
        return combined_df
    else:
        logger.warning("No currency files found to combine")
        return None

def validate_data(currency_dir: str):
    """
    Validate the processed currency data.
    
    Args:
        currency_dir: Directory containing processed currency files
    
    Returns:
        Dictionary with validation results
    """
    
    logger.info("Validating processed data...")
    
    validation_results = {
        'total_files': 0,
        'valid_files': 0,
        'errors': [],
        'warnings': []
    }
    
    # Get all currency CSV files
    currency_files = list(Path(currency_dir).glob("*.csv"))
    currency_files = [f for f in currency_files if f.name != "currency_summary.csv"]
    
    validation_results['total_files'] = len(currency_files)
    
    for file_path in currency_files:
        try:
            # Read the file
            df = pd.read_csv(file_path)
            
            # Check required columns
            required_columns = ['date', 'rate']
            if not all(col in df.columns for col in required_columns):
                validation_results['errors'].append(f"{file_path.name}: Missing required columns")
                continue
            
            # Check for NaN values in rate column
            nan_count = df['rate'].isna().sum()
            if nan_count > 0:
                validation_results['warnings'].append(f"{file_path.name}: {nan_count} NaN values in rate column")
            
            # Check date format (basic check)
            try:
                pd.to_datetime(df['date'])
            except:
                validation_results['warnings'].append(f"{file_path.name}: Date format issues")
            
            # Check for duplicates
            duplicate_dates = df['date'].duplicated().sum()
            if duplicate_dates > 0:
                validation_results['warnings'].append(f"{file_path.name}: {duplicate_dates} duplicate dates")
            
            # Check data range
            if len(df) < 10:
                validation_results['warnings'].append(f"{file_path.name}: Very few data points ({len(df)})")
            
            validation_results['valid_files'] += 1
            
        except Exception as e:
            validation_results['errors'].append(f"{file_path.name}: Read error - {str(e)}")
    
    # Log validation results
    if validation_results['errors']:
        logger.error(f"Validation found {len(validation_results['errors'])} errors")
        for error in validation_results['errors']:
            logger.error(f"  {error}")
    else:
        logger.info("No validation errors found")
    
    if validation_results['warnings']:
        logger.warning(f"Validation found {len(validation_results['warnings'])} warnings")
        for warning in validation_results['warnings']:
            logger.warning(f"  {warning}")
    
    return validation_results

def main():
    """Main function for data preparation script."""
    
    parser = argparse.ArgumentParser(description='Prepare exchange rate data for forecasting.')
    parser.add_argument('--input', type=str, default='data/raw/exchange_rate_usd_to.csv',
                       help='Path to raw CSV file with all currencies')
    parser.add_argument('--output-dir', type=str, default='data/processed/currencies',
                       help='Directory to save individual currency files')
    parser.add_argument('--validate', action='store_true',
                       help='Validate the processed data')
    parser.add_argument('--create-combined', action='store_true',
                       help='Create a combined dataset from individual files')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Check if input file exists
    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        return
    
    logger.info("=" * 60)
    logger.info("Exchange Rate Data Preparation")
    logger.info("=" * 60)
    
    # Step 1: Split the raw data into individual currency files
    try:
        stats = split_currency_data(args.input, args.output_dir)
        
        logger.info(f"Successfully processed {stats['processed_currencies']} out of {stats['total_currencies']} currencies")
        
        # Step 2: Create combined dataset if requested
        if args.create_combined:
            combined_output = Path(args.output_dir).parent / "combined_currencies.csv"
            create_combined_dataset(args.output_dir, str(combined_output))
        
        # Step 3: Validate data if requested
        if args.validate:
            validation_results = validate_data(args.output_dir)
            
            if validation_results['valid_files'] == validation_results['total_files']:
                logger.info("✅ All files validated successfully")
            else:
                logger.warning(f"⚠️  {validation_results['valid_files']}/{validation_results['total_files']} files valid")
        
        logger.info("=" * 60)
        logger.info("Data preparation completed successfully!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Data preparation failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())

