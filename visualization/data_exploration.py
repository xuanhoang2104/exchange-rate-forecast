"""
Script khám phá dữ liệu tỷ giá hối đoái
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Cấu hình đường dẫn
INPUT_DIR = r"D:\FPT\ki 7\DAT\project_exchange_rate\exchange-rate-forecast\data\processed\currencies"
OUTPUT_BASE_DIR = r"D:\FPT\kì 7\DAT\exchange-rate\data\processed"

def create_output_directories():
    """Tạo các thư mục đầu ra nếu chưa tồn tại"""
    directories = [
        os.path.join(OUTPUT_BASE_DIR, "merged"),
        os.path.join(OUTPUT_BASE_DIR, "..", "..", "outputs", "figures", "eda"),
        os.path.join(OUTPUT_BASE_DIR, "..", "..", "logs")
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Da tao thu muc: {directory}")

def merge_currency_data(currency_dir):
    """
    Gop tat ca cac file currency thanh mot DataFrame
    """
    print("\n" + "="*70)
    print("GOP DU LIEU CAC LOAI TIEN")
    print("="*70)
    
    currency_files = []
    all_data = {}
    
    # Doc tat ca cac file CSV trong thu muc currencies
    for file_name in os.listdir(currency_dir):
        if file_name.endswith('.csv') and file_name != 'currency_summary.csv' and file_name != 'detailed_statistics.csv':
            currency_name = file_name.replace('.csv', '')
            file_path = os.path.join(currency_dir, file_name)
            
            try:
                df = pd.read_csv(file_path, parse_dates=['date'])
                if 'rate' in df.columns and len(df) > 0:
                    # Kiem tra xem du lieu co thay doi khong
                    if df['rate'].nunique() > 1:  # Co it nhat 2 gia tri khac nhau
                        all_data[currency_name] = df.set_index('date')['rate']
                        currency_files.append(currency_name)
                        print(f"  Da doc: {currency_name} ({len(df)} dong, {df['rate'].nunique()} gia tri khac nhau)")
                    else:
                        print(f"  Bo qua: {currency_name} (du lieu khong thay doi)")
                else:
                    print(f"  Canh bao: {file_name} khong co cot 'rate'")
            except Exception as e:
                print(f"  Loi khi doc {file_name}: {e}")
    
    if not all_data:
        print("Khong doc duoc file nao")
        return None
    
    # Gop tat ca thanh mot DataFrame
    merged_df = pd.concat(all_data, axis=1)
    merged_df.columns = [col[0] for col in merged_df.columns] if isinstance(merged_df.columns, pd.MultiIndex) else merged_df.columns
    
    print(f"\nTong so loai tien: {len(merged_df.columns)}")
    print(f"So ngay du lieu: {len(merged_df)}")
    print(f"Tu ngay: {merged_df.index.min().date()} den {merged_df.index.max().date()}")
    
    # Luu file da gop
    merged_path = os.path.join(OUTPUT_BASE_DIR, "merged", "all_currencies.csv")
    merged_df.to_csv(merged_path)
    print(f"\nDa luu file da gop tai: {merged_path}")
    
    return merged_df

def load_and_explore_data(df):
    """
    Kham pha du lieu
    """
    print("\n" + "="*70)
    print("KHAI PHAP DU LIEU BAN DAU")
    print("="*70)
    
    print(f"Kich thuoc du lieu: {df.shape}")
    print(f"Thoi gian: {df.index.min()} den {df.index.max()}")
    print(f"So luong loai tien: {len(df.columns)}")
    
    # Thong tin co ban
    print("\nThong tin co ban:")
    print(df.info())
    
    # Thong ke mo ta
    print("\nThong ke mo ta:")
    print(df.describe())
    
    # Kiem tra missing values
    print("\nKiem tra missing values:")
    missing_stats = df.isnull().sum()
    missing_percentage = (missing_stats / len(df) * 100)
    
    missing_df = pd.DataFrame({
        'missing_count': missing_stats,
        'missing_percentage': missing_percentage
    })
    
    print(missing_df[missing_df['missing_count'] > 0])
    
    # Kiem tra du lieu constant
    print("\nKiem tra du lieu khong thay doi (constant):")
    constant_currencies = []
    for col in df.columns:
        if df[col].dropna().nunique() <= 1:
            constant_currencies.append(col)
    
    if constant_currencies:
        print(f"Co {len(constant_currencies)} loai tien co du lieu khong thay doi:")
        for curr in constant_currencies:
            print(f"  - {curr}")
    else:
        print("Khong co loai tien nao co du lieu khong thay doi")
    
    return df

def visualize_currency_series(df, currencies=None, n_currencies=8):
    """
    Truc quan hoa chuoi thoi gian cua cac loai tien
    """
    if df is None or len(df) == 0:
        print("Khong co du lieu de truc quan hoa")
        return
    
    if currencies is None:
        # Chon ngau nhien n_currencies de hien thi
        available_currencies = df.columns.tolist()
        n_currencies = min(n_currencies, len(available_currencies))
        currencies = np.random.choice(available_currencies, n_currencies, replace=False)
    
    fig, axes = plt.subplots(n_currencies, 1, figsize=(15, 3*n_currencies))
    if n_currencies == 1:
        axes = [axes]
    
    for idx, currency in enumerate(currencies):
        ax = axes[idx]
        
        # Lay du lieu khong bi NaN
        series = df[currency].dropna()
        if len(series) == 0:
            ax.text(0.5, 0.5, f'No data for {currency}', 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes)
            ax.set_title(f'{currency} - No Data')
            ax.axis('off')
            continue
            
        series.plot(ax=ax, title=f'{currency} Exchange Rate to USD', linewidth=1)
        ax.set_ylabel('Rate')
        ax.grid(True, alpha=0.3)
        
        # Them annotation cho gia tri dau va cuoi
        if len(series) > 0:
            first_val = series.iloc[0]
            last_val = series.iloc[-1]
            change_pct = ((last_val - first_val) / first_val) * 100
            
            ax.text(0.02, 0.95, f'First: {first_val:.2f}\nLast: {last_val:.2f}\nChange: {change_pct:+.1f}%',
                    transform=ax.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Luu hinh
    output_file = os.path.join(OUTPUT_BASE_DIR, "..", "..", "outputs", "figures", "eda", "currency_series.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Da luu hinh tai: {output_file}")
    plt.show()

def safe_adfuller_test(series):
    """
    Thuc hien ADF test mot cach an toan, tranh loi voi du lieu constant
    """
    try:
        # Kiem tra xem series co du du lieu va khong phai constant khong
        if len(series.dropna()) < 10:
            return None, "Insufficient data (less than 10 observations)"
        
        if series.dropna().nunique() <= 1:
            return None, "Constant data (no variation)"
        
        # Thuc hien ADF test
        result = adfuller(series.dropna())
        return result, "Success"
        
    except Exception as e:
        return None, f"ADF test error: {str(e)}"

def check_stationarity(df, currencies=None, n_check=5):
    """
    Kiem tra tinh dung cua chuoi thoi gian su dung ADF test
    """
    if df is None or len(df) == 0:
        print("Khong co du lieu de kiem tra")
        return pd.DataFrame()
    
    if currencies is None:
        available_currencies = df.columns.tolist()
        n_check = min(n_check, len(available_currencies))
        currencies = np.random.choice(available_currencies, n_check, replace=False)
    
    results = []
    for currency in currencies:
        series = df[currency]
        
        # Thuc hien ADF test an toan
        result, message = safe_adfuller_test(series)
        
        if result is None:
            print(f"\n{currency}:")
            print(f"  Khong the kiem tra: {message}")
            continue
            
        # Luu ket qua
        results.append({
            'currency': currency,
            'adf_statistic': result[0],
            'p_value': result[1],
            'critical_value_1%': result[4]['1%'],
            'critical_value_5%': result[4]['5%'],
            'critical_value_10%': result[4]['10%'],
            'is_stationary': result[1] < 0.05,
            'test_message': message
        })
        
        print(f"\n{currency}:")
        print(f"  ADF Statistic: {result[0]:.4f}")
        print(f"  p-value: {result[1]:.4f}")
        print(f"  La chuoi dung: {result[1] < 0.05}")
    
    if results:
        return pd.DataFrame(results)
    else:
        print("Khong co loai tien nao co the kiem tra ADF")
        return pd.DataFrame()

def correlation_analysis(df):
    """
    Phan tuong quan giua cac loai tien
    """
    if df is None or len(df) < 2:
        print("Khong du du lieu de phan tich tuong quan")
        return None, []
    
    # Loai bo cac cot co qua nhieu missing values (>50%)
    df_clean = df.dropna(thresh=len(df)*0.5, axis=1)
    
    if len(df_clean.columns) < 2:
        print("Khong du loai tien de phan tich tuong quan sau khi loai bo missing values")
        return None, []
    
    print(f"Phan tich tuong quan cho {len(df_clean.columns)} loai tien")
    
    # Tinh ma tran tuong quan
    corr_matrix = df_clean.corr()
    
    # Ve heatmap
    plt.figure(figsize=(max(12, len(df_clean.columns)), max(10, len(df_clean.columns)*0.8)))
    sns.heatmap(corr_matrix, 
                cmap='coolwarm', 
                center=0,
                square=True,
                cbar_kws={"shrink": 0.8},
                annot=False,
                linewidths=0.5)
    plt.title('Correlation Matrix of Exchange Rates', fontsize=16, pad=20)
    plt.tight_layout()
    
    # Luu hinh
    output_file = os.path.join(OUTPUT_BASE_DIR, "..", "..", "outputs", "figures", "eda", "correlation_matrix.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Da luu hinh tai: {output_file}")
    plt.show()
    
    # Tim cac cap tuong quan cao
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_value = corr_matrix.iloc[i, j]
            if not pd.isna(corr_value) and abs(corr_value) > 0.7:
                high_corr_pairs.append({
                    'currency1': corr_matrix.columns[i],
                    'currency2': corr_matrix.columns[j],
                    'correlation': corr_value
                })
    
    # Sap xep theo correlation tuyet doi
    high_corr_pairs.sort(key=lambda x: abs(x['correlation']), reverse=True)
    
    print(f"\nTim thay {len(high_corr_pairs)} cap co |correlation| > 0.7")
    for pair in high_corr_pairs[:15]:
        symbol = "+" if pair['correlation'] > 0 else "-"
        print(f"  {pair['currency1']} - {pair['currency2']}: {pair['correlation']:+.3f}")
    
    return corr_matrix, high_corr_pairs

def return_analysis(df):
    """
    Phan tich loi suat (returns)
    """
    if df is None or len(df) < 2:
        print("Khong du du lieu de phan tich returns")
        return None
    
    # Loai bo cac cot co qua nhieu missing values
    df_clean = df.dropna(thresh=len(df)*0.3, axis=1)
    
    if len(df_clean.columns) == 0:
        print("Khong co loai tien nao du du lieu de phan tich returns")
        return None
    
    # Tinh daily returns
    returns = df_clean.pct_change().dropna()
    
    if returns.empty or len(returns.columns) == 0:
        print("Khong the tinh returns")
        return None
    
    print(f"Phan tich returns cho {len(returns.columns)} loai tien")
    
    # Ve phan phoi returns
    n_plots = min(6, len(returns.columns))
    selected_currencies = returns.columns[:n_plots].tolist()
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx in range(6):
        ax = axes[idx]
        
        if idx < n_plots:
            currency = selected_currencies[idx]
            returns[currency].dropna().hist(bins=50, ax=ax, alpha=0.7, color='skyblue', edgecolor='black')
            ax.set_title(f'{currency} Returns Distribution', fontsize=10)
            ax.set_xlabel('Daily Return')
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)
            
            # Them thong ke
            mean_return = returns[currency].mean()
            std_return = returns[currency].std()
            
            ax.axvline(mean_return, color='red', linestyle='--', linewidth=2, 
                      label=f'Mean: {mean_return:.4f}')
            ax.axvline(mean_return + std_return, color='orange', linestyle=':', linewidth=1.5)
            ax.axvline(mean_return - std_return, color='orange', linestyle=':', linewidth=1.5)
            
            # Them text box voi thong ke
            stats_text = f'Mean: {mean_return:.4f}\nStd: {std_return:.4f}\nSkew: {returns[currency].skew():.2f}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   verticalalignment='top', fontsize=8,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            ax.legend(fontsize=8)
        else:
            ax.axis('off')
    
    plt.tight_layout()
    
    # Luu hinh
    output_file = os.path.join(OUTPUT_BASE_DIR, "..", "..", "outputs", "figures", "eda", "returns_distribution.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Da luu hinh tai: {output_file}")
    plt.show()
    
    return returns

def analyze_top_currencies(df, top_n=10):
    """
    Phan tich cac loai tien co nhieu du lieu nhat
    """
    if df is None:
        return None
    
    # Dem so ngay co du lieu cho moi loai tien
    currency_stats = pd.DataFrame({
        'currency': df.columns,
        'data_points': df.notna().sum().values,
        'missing_percentage': (df.isna().sum() / len(df) * 100).values,
        'unique_values': [df[col].dropna().nunique() for col in df.columns],
        'is_constant': [df[col].dropna().nunique() <= 1 for col in df.columns],
        'mean_rate': df.mean(),
        'std_rate': df.std(),
        'min_rate': df.min(),
        'max_rate': df.max()
    })
    
    # Sap xep theo so luong du lieu
    currency_stats = currency_stats.sort_values('data_points', ascending=False)
    
    print("\n" + "="*70)
    print(f"TOP {top_n} LOAI TIEN CO NHIEU DU LIEU NHAT")
    print("="*70)
    
    print(currency_stats.head(top_n).to_string())
    
    # Ve bieu do
    plt.figure(figsize=(12, 6))
    top_currencies = currency_stats.head(top_n)
    bars = plt.bar(range(len(top_currencies)), top_currencies['data_points'])
    plt.xlabel('Currency')
    plt.ylabel('Number of Data Points')
    plt.title(f'Top {top_n} Currencies by Data Availability')
    plt.xticks(range(len(top_currencies)), top_currencies['currency'], rotation=45, ha='right')
    
    # Them so luong len tren moi cot
    for bar, count in zip(bars, top_currencies['data_points']):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                f'{int(count)}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    # Luu hinh
    output_file = os.path.join(OUTPUT_BASE_DIR, "..", "..", "outputs", "figures", "eda", "top_currencies.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nDa luu hinh tai: {output_file}")
    plt.show()
    
    return currency_stats

def analyze_currency_characteristics(df):
    """
    Phan tich dac trung cua cac loai tien
    """
    if df is None:
        return
    
    print("\n" + "="*70)
    print("PHAN TICH DAC TRUNG CAC LOAI TIEN")
    print("="*70)
    
    # Phan loai theo vung/loai tien
    currency_categories = {
        'europe': ['euro', 'british_pound', 'swiss_franc', 'swedish_krona', 'norwegian_krone', 
                  'danish_krone', 'czech_koruna', 'hungarian_forint', 'polish_zloty'],
        'asia': ['chinese_yuan', 'japanese_yen', 'singapore_dollar', 'hong_kong_dollar',
                'south_korean_won', 'indian_rupee', 'indonesian_rupiah', 'philippine_peso',
                'thai_baht', 'malaysian_ringgit', 'vietnamese_dong'],
        'america': ['us_dollar', 'canadian_dollar', 'mexican_peso', 'brazilian_real',
                   'argentine_peso', 'chilean_peso', 'colombian_peso'],
        'middle_east': ['saudi_riyal', 'uae_dirham', 'qatari_riyal', 'kuwaiti_dinar',
                       'omani_rial', 'bahrain_dinar', 'iranian_rial'],
        'other': []  # Cac loai tien con lai
    }
    
    # Dem so luong tien theo vung
    for category, currencies in currency_categories.items():
        if category != 'other':
            found = [curr for curr in currencies if curr in df.columns]
            print(f"{category.title()}: {len(found)} loai tien")
            if found:
                print(f"  {', '.join(found)}")
    
    # Phan tich missing data theo thoi gian
    print("\nPhan tich missing data theo thoi gian:")
    missing_by_date = df.isnull().sum(axis=1)
    
    plt.figure(figsize=(12, 4))
    missing_by_date.plot()
    plt.title('Number of Missing Currencies Over Time')
    plt.xlabel('Date')
    plt.ylabel('Number of Missing Currencies')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_file = os.path.join(OUTPUT_BASE_DIR, "..", "..", "outputs", "figures", "eda", "missing_over_time.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Da luu hinh tai: {output_file}")
    plt.show()

def generate_summary_report(df, stationarity_df, high_corr_pairs, returns, currency_stats):
    """
    Tao bao cao tong hop
    """
    if df is None:
        return {}
    
    report = {
        'total_currencies': len(df.columns),
        'total_observations': len(df),
        'date_range_start': df.index.min().strftime('%Y-%m-%d'),
        'date_range_end': df.index.max().strftime('%Y-%m-%d'),
        'total_days': (df.index.max() - df.index.min()).days,
        'missing_values_count': df.isnull().sum().sum(),
        'missing_percentage': (df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100),
    }
    
    if currency_stats is not None and not currency_stats.empty:
        report['avg_data_points'] = currency_stats['data_points'].mean()
        report['median_data_points'] = currency_stats['data_points'].median()
        report['top_currency'] = currency_stats.iloc[0]['currency']
        report['top_currency_points'] = int(currency_stats.iloc[0]['data_points'])
        report['constant_currencies'] = currency_stats['is_constant'].sum()
    
    if stationarity_df is not None and not stationarity_df.empty:
        report['stationary_currencies'] = stationarity_df['is_stationary'].sum()
        report['non_stationary_currencies'] = len(stationarity_df) - stationarity_df['is_stationary'].sum()
    
    if high_corr_pairs:
        report['high_correlation_pairs'] = len(high_corr_pairs)
    
    if returns is not None and not returns.empty:
        report['avg_daily_return'] = returns.mean().mean()
        report['avg_volatility'] = returns.std().mean()
        report['avg_skewness'] = returns.skew().mean()
        report['avg_kurtosis'] = returns.kurtosis().mean()
    
    return report

def main():
    """Chay toan bo qua trinh kham pha du lieu"""
    print("BAT DAU KHAM PHA DU LIEU")
    print("="*70)
    print(f"Input directory: {INPUT_DIR}")
    print(f"Output base directory: {OUTPUT_BASE_DIR}")
    print("="*70)
    
    # Kiem tra thu muc input
    if not os.path.exists(INPUT_DIR):
        print(f"Thu muc input khong ton tai: {INPUT_DIR}")
        print("Vui long kiem tra lai duong dan")
        return
    
    # 1. Tao thu muc dau ra
    create_output_directories()
    
    # 2. Gop du lieu cac loai tien
    df = merge_currency_data(INPUT_DIR)
    
    if df is None:
        print("Khong the tiep tuc do thieu du lieu")
        return
    
    # 3. Kham pha du lieu
    print("\n" + "="*70)
    print("KHAI PHAP DU LIEU")
    print("="*70)
    df = load_and_explore_data(df)
    
    # 4. Phan tich dac trung cac loai tien
    analyze_currency_characteristics(df)
    
    # 5. Phan tich cac loai tien co nhieu du lieu nhat
    currency_stats = analyze_top_currencies(df, top_n=15)
    
    # 6. Truc quan hoa chuoi thoi gian (chon 6 loai tien co nhieu du lieu nhat)
    print("\n" + "="*70)
    print("TRUC QUAN HOA CHUOI THOI GIAN")
    print("="*70)
    
    if currency_stats is not None:
        # Chon cac loai tien co nhieu du lieu va khong phai constant
        top_currencies = currency_stats[~currency_stats['is_constant']].head(6)['currency'].tolist()
        if top_currencies:
            visualize_currency_series(df, currencies=top_currencies)
        else:
            print("Khong co loai tien nao co du lieu thay doi de hien thi")
    
    # 7. Kiem tra tinh dung
    print("\n" + "="*70)
    print("KIEM TRA TINH DUNG (ADF TEST)")
    print("="*70)
    
    # Chon 8 loai tien co nhieu du lieu nhat va khong phai constant
    if currency_stats is not None:
        test_currencies = currency_stats[~currency_stats['is_constant']].head(8)['currency'].tolist()
        stationarity_df = check_stationarity(df, currencies=test_currencies)
    else:
        stationarity_df = pd.DataFrame()
    
    # 8. Phan tich tuong quan
    print("\n" + "="*70)
    print("PHAN TICH TUONG QUAN")
    print("="*70)
    corr_matrix, high_corr_pairs = correlation_analysis(df)
    
    # 9. Phan tich returns (chon cac loai tien co nhieu du lieu)
    print("\n" + "="*70)
    print("PHAN TICH LOI SUAT (RETURNS)")
    print("="*70)
    
    if currency_stats is not None:
        # Chon cac loai tien co it nhat 50 ngay du lieu va khong phai constant
        currencies_with_data = currency_stats[
            (currency_stats['data_points'] >= 50) & 
            (~currency_stats['is_constant'])
        ]['currency'].tolist()
        
        if currencies_with_data:
            df_filtered = df[currencies_with_data].dropna(thresh=50)  # It nhat 50 ngay co du lieu
            if len(df_filtered.columns) > 0:
                returns = return_analysis(df_filtered)
            else:
                returns = None
                print("Khong du loai tien co du du lieu de phan tich returns")
        else:
            returns = None
            print("Khong co loai tien nao co du 50 ngay du lieu thay doi")
    else:
        returns = None
    
    # 10. Tao bao cao tong hop
    print("\n" + "="*70)
    print("BAO CAO TONG HOP")
    print("="*70)
    
    report = generate_summary_report(df, stationarity_df, high_corr_pairs, returns, currency_stats)
    
    for key, value in report.items():
        print(f"  {key.replace('_', ' ').title()}: {value}")
    
    # 11. Luu cac file xu ly
    try:
        # Luu thong ke currency
        if currency_stats is not None:
            stats_path = os.path.join(OUTPUT_BASE_DIR, "merged", "currency_statistics.csv")
            currency_stats.to_csv(stats_path, index=False)
            print(f"\nDa luu thong ke currency tai: {stats_path}")
        
        # Luu correlation matrix
        if corr_matrix is not None:
            corr_path = os.path.join(OUTPUT_BASE_DIR, "merged", "correlation_matrix.csv")
            corr_matrix.to_csv(corr_path)
            print(f"Da luu correlation matrix tai: {corr_path}")
        
        # Luu returns
        if returns is not None:
            returns_path = os.path.join(OUTPUT_BASE_DIR, "merged", "daily_returns.csv")
            returns.to_csv(returns_path)
            print(f"Da luu daily returns tai: {returns_path}")
            
    except Exception as e:
        print(f"Loi khi luu du lieu: {e}")
    
    # 12. Luu bao cao ra file text
    try:
        log_dir = os.path.join(OUTPUT_BASE_DIR, "..", "..", "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "eda_report.txt")
        
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write("BAO CAO KHAM PHA DU LIEU TY GIA\n")
            f.write("="*50 + "\n\n")
            f.write(f"Thoi gian chay: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Thu muc du lieu: {INPUT_DIR}\n\n")
            
            for key, value in report.items():
                f.write(f"{key.replace('_', ' ').title()}: {value}\n")
            
            f.write("\n" + "="*50 + "\n")
            f.write("TOP 10 CURRENCIES BY DATA AVAILABILITY\n")
            f.write("="*50 + "\n")
            if currency_stats is not None:
                f.write(currency_stats.head(10).to_string())
        
        print(f"\nDa luu bao cao tai: {log_file}")
    except Exception as e:
        print(f"Khong luu duoc bao cao text: {e}")
    
    print("\n" + "="*70)
    print("HOAN THANH KHAM PHA DU LIEU!")
    print("="*70)
    print("Cac file da duoc luu tai:")
    print(f"  - {os.path.join(OUTPUT_BASE_DIR, 'merged')}")
    print(f"  - {os.path.join(OUTPUT_BASE_DIR, '..', '..', 'outputs', 'figures', 'eda')}")
    print(f"  - {os.path.join(OUTPUT_BASE_DIR, '..', '..', 'logs')}")

if __name__ == "__main__":
    main()

"""
GIẢI THÍCH DỮ LIỆU:
1. Phạm vi thời gian rộng
Từ: 2004-01-02 đến 2026-01-30 (22 năm)

Tổng số ngày: 5573 ngày làm việc (không tính cuối tuần)

Đây là dữ liệu lịch sử dài hạn rất quý giá

2. Số lượng loại tiền
Tổng: 47 loại tiền khác nhau

Đã bỏ qua 5 loại tiền có dữ liệu không thay đổi (constant):

bahrain_dinar

omani_rial

qatari_riyal

saudi_arabian_riyal

us_dollar (luôn = 1.0 vì là đồng tiền tham chiếu)

PHÂN TÍCH CHẤT LƯỢNG DỮ LIỆU
3. Missing data đáng kể
Tổng missing values: 53,298 (20.35% tổng số ô dữ liệu)

Các loại tiền có nhiều missing data nhất:

vnd: 91.05% missing (chỉ có 499 ngày dữ liệu)

tunisian_dinar: 66.27% missing

bolivar_fuerte: 57.26% missing

4. Top 15 loại tiền có nhiều dữ liệu nhất:
Euro: 5,469 ngày (98.1% coverage)

UK Pound: 5,442 ngày (97.7%)

Swiss Franc: 5,329 ngày (95.6%)

Czech Koruna: 5,314 ngày (95.4%)

Polish Zloty: 5,310 ngày (95.3%)

PHÂN TÍCH THỐNG KÊ QUAN TRỌNG
5. Tính dừng (Stationarity)
KẾT LUẬN: Không có loại tiền nào là chuỗi dừng (p-value > 0.05 cho tất cả)

Ý nghĩa: Tất cả tỷ giá đều có xu hướng (trend) và cần phải làm cho dừng trước khi xây dựng mô hình

6. Tương quan (Correlation) cực cao
451 cặp có |correlation| > 0.7

Các cặp tương quan hoàn hảo (≈1.0):

brunei_dollar - singapore_dollar: +1.000

danish_krone - euro: +1.000

indian_rupee - nepalese_rupee: +1.000

Giải thích:

Brunei Dollar và Singapore Dollar luôn có tỷ giá giống hệt nhau

Danish Krone và Euro cố định tỷ giá (EUR/DKK ≈ 7.46)

Đây là các cặp tiền tệ có chính sách tỷ giá cố định

7. Returns analysis không thực hiện được
Lý do: Dữ liệu có quá nhiều missing values, không thể tính daily returns đầy đủ

PHÂN TÍCH THEO VÙNG ĐỊA LÝ
8. Phân bố theo khu vực:
Châu Âu: 8 loại tiền (Euro, GBP, CHF, SEK, NOK, DKK, CZK, HUF, PLN)

Châu Á: 8 loại tiền (CNY, JPY, SGD, INR, IDR, PHP, THB, MYR)

Châu Mỹ: 5 loại tiền (CAD, MXN, BRL, CLP, COP)

Trung Đông: 3 loại tiền (AED, KWD, IRR)

VẤN ĐỀ VỚI DỮ LIỆU VND
9. Dữ liệu VND rất thiếu
Chỉ có: 499 ngày dữ liệu (8.9% coverage)

Phạm vi: Không rõ từ thời gian nào

Giá trị trung bình: 25,662 VND/USD

Biến động: 543 VND/USD

ĐỀ XUẤT CHO DỰ ÁN
10. Xử lý trước khi modeling:
Lọc loại tiền: Giữ lại 15-20 loại tiền có nhiều dữ liệu nhất

Xử lý missing data:

Interpolation cho các missing ngắn

Drop các loại tiền có quá nhiều missing (>30%)

Làm cho dữ liệu dừng: Differencing hoặc transformation

Xử lý multicollinearity: Loại bỏ các cặp tương quan hoàn hảo

11. Dữ liệu VND cần được bổ sung:
Cần thu thập thêm dữ liệu VND từ 2020 trở đi

Hiện tại quá ít dữ liệu để xây dựng mô hình tốt

KẾT LUẬN
Điểm mạnh:

Dữ liệu lịch sử dài (22 năm)

Nhiều loại tiền đa dạng

Top currencies có coverage rất tốt

Điểm yếu:

Missing data nhiều

VND có quá ít dữ liệu

Nhiều cặp tiền tương quan hoàn hảo (cần xử lý)

Tất cả chuỗi đều không dừng

Khuyến nghị: Tập trung vào top 10-15 currencies có nhiều dữ liệu nhất, bổ sung thêm dữ liệu VND trước khi xây dựng mô hình dự báo.



"""