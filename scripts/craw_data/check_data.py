#!/usr/bin/env python3
"""
Liệt kê tất cả các đồng tiền trong thư mục currencies
"""
import pandas as pd
from pathlib import Path
import re

def list_all_currencies():
    """Liệt kê tất cả đồng tiền trong thư mục"""
    
    # Đường dẫn thư mục
    currency_dir = Path("D:/FPT/ki 7/DAT/project_exchange_rate/exchange-rate-forecast/data/processed/currencies")
    
    if not currency_dir.exists():
        print(f"❌ Thư mục không tồn tại: {currency_dir}")
        return
    
    # Lấy tất cả file CSV
    csv_files = list(currency_dir.glob("*.csv"))
    
    print(f"📁 Tổng số file CSV: {len(csv_files)}")
    print("=" * 80)
    
    # Phân loại file
    summary_files = [f for f in csv_files if "summary" in f.name.lower()]
    currency_files = [f for f in csv_files if "summary" not in f.name.lower()]
    
    print(f"📊 File tổng hợp: {len(summary_files)}")
    for f in summary_files:
        print(f"  • {f.name}")
    
    print(f"\n💰 File đồng tiền riêng: {len(currency_files)}")
    
    # Đọc file summary nếu có
    summary_data = []
    summary_file = currency_dir / "currency_summary.csv"
    
    if summary_file.exists():
        print(f"\n📋 Đọc thông tin từ file summary...")
        try:
            df_summary = pd.read_csv(summary_file)
            print(f"  Tìm thấy {len(df_summary)} đồng tiền trong summary")
            
            # Hiển thị tất cả đồng tiền
            print("\n🎯 DANH SÁCH ĐẦY ĐỦ CÁC ĐỒNG TIỀN:")
            print("=" * 80)
            
            for i, (_, row) in enumerate(df_summary.iterrows(), 1):
                currency_name = row['currency']
                data_points = row['data_points']
                first_date = row['first_date']
                last_date = row['last_date']
                
                print(f"{i:3d}. {currency_name:30s} - {data_points:4d} dòng ({first_date} đến {last_date})")
            
            summary_data = df_summary.to_dict('records')
            
        except Exception as e:
            print(f"  Lỗi đọc summary: {e}")
    
    # Nếu không có summary, liệt kê từ file
    if not summary_data:
        print("\n📋 LIỆT KÊ TỪ FILE:")
        print("=" * 80)
        
        # Phân loại theo khu vực
        currency_categories = {
            'CHÂU Á': [],
            'CHÂU ÂU': [],
            'CHÂU MỸ': [],
            'TRUNG ĐÔNG & CHÂU PHI': [],
            'CHÂU ĐẠI DƯƠNG': [],
            'KHÁC': []
        }
        
        # Map tên file -> tên quốc gia
        country_map = {
            'chinese_yuan': 'Trung Quốc',
            'japanese_yen': 'Nhật Bản',
            'indian_rupee': 'Ấn Độ',
            'korean_won': 'Hàn Quốc',
            'singapore_dollar': 'Singapore',
            'thai_baht': 'Thái Lan',
            'malaysian_ringgit': 'Malaysia',
            'indonesian_rupiah': 'Indonesia',
            'philippine_peso': 'Philippines',
            'vietnamese_dong': 'Việt Nam',
            'taiwan_dollar': 'Đài Loan',
            'hong_kong_dollar': 'Hong Kong',
            
            'euro': 'Eurozone',
            'uk_pound': 'Anh',
            'swiss_franc': 'Thụy Sĩ',
            'swedish_krona': 'Thụy Điển',
            'norwegian_krone': 'Na Uy',
            'danish_krone': 'Đan Mạch',
            'polish_zloty': 'Ba Lan',
            'czech_koruna': 'Séc',
            'hungarian_forint': 'Hungary',
            'russian_ruble': 'Nga',
            
            'canadian_dollar': 'Canada',
            'mexican_peso': 'Mexico',
            'brazilian_real': 'Brazil',
            'argentine_peso': 'Argentina',
            'chilean_peso': 'Chile',
            'colombian_peso': 'Colombia',
            'peruvian_sol': 'Peru',
            
            'saudi_arabian_riyal': 'Ả Rập Xê Út',
            'uae_dirham': 'UAE',
            'kuwaiti_dinar': 'Kuwait',
            'qatari_riyal': 'Qatar',
            'iranian_rial': 'Iran',
            'israeli_new_shekel': 'Israel',
            'turkish_lira': 'Thổ Nhĩ Kỳ',
            
            'australian_dollar': 'Australia',
            'new_zealand_dollar': 'New Zealand'
        }
        
        for i, file in enumerate(sorted(currency_files), 1):
            file_name = file.stem.lower()
            
            # Tìm quốc gia tương ứng
            country_name = "N/A"
            for key, value in country_map.items():
                if key in file_name:
                    country_name = value
                    break
            
            # Phân loại khu vực
            if any(x in file_name for x in ['chinese', 'japanese', 'indian', 'korean', 'singapore', 
                                           'thai', 'malaysian', 'indonesian', 'philippine', 'vietnamese']):
                region = 'CHÂU Á'
            elif any(x in file_name for x in ['euro', 'pound', 'swiss', 'swedish', 'norwegian', 
                                             'danish', 'polish', 'czech', 'hungarian', 'russian']):
                region = 'CHÂU ÂU'
            elif any(x in file_name for x in ['canadian', 'mexican', 'brazilian', 'argentine', 
                                             'chilean', 'colombian', 'peruvian']):
                region = 'CHÂU MỸ'
            elif any(x in file_name for x in ['saudi', 'uae', 'kuwaiti', 'qatari', 'iranian', 
                                             'israeli', 'turkish']):
                region = 'TRUNG ĐÔNG & CHÂU PHI'
            elif any(x in file_name for x in ['australian', 'zealand']):
                region = 'CHÂU ĐẠI DƯƠNG'
            else:
                region = 'KHÁC'
            
            # Đếm số dòng
            try:
                df = pd.read_csv(file)
                row_count = len(df)
                
                # Thêm vào danh sách phân loại
                currency_info = {
                    'file': file.name,
                    'country': country_name,
                    'currency': file_name.replace('_', ' ').title(),
                    'rows': row_count
                }
                currency_categories[region].append(currency_info)
                
                print(f"{i:3d}. {file_name:30s} - {row_count:4d} dòng ({country_name})")
                
            except Exception as e:
                print(f"{i:3d}. {file_name:30s} - Lỗi đọc file")
        
        # Hiển thị phân loại
        print("\n" + "=" * 80)
        print("🏳️‍🌈 PHÂN LOẠI THEO KHU VỰC:")
        print("=" * 80)
        
        total_currencies = 0
        for region, currencies in currency_categories.items():
            if currencies:
                print(f"\n{region}:")
                print("-" * 40)
                for curr in sorted(currencies, key=lambda x: x['country']):
                    print(f"  • {curr['country']:20s} - {curr['currency']:25s} ({curr['rows']} dòng)")
                total_currencies += len(currencies)
        
        print(f"\n📊 Tổng cộng: {total_currencies} đồng tiền từ {len(currency_categories)} khu vực")
    
    # Tạo file thống kê
    create_statistics_file(currency_dir, currency_files)

def create_statistics_file(currency_dir, currency_files):
    """Tạo file thống kê chi tiết"""
    print("\n" + "=" * 80)
    print("📈 TẠO FILE THỐNG KÊ CHI TIẾT...")
    print("=" * 80)
    
    stats_data = []
    
    for file in sorted(currency_files):
        try:
            df = pd.read_csv(file)
            file_name = file.stem
            row_count = len(df)
            
            # Lấy thông tin ngày
            if 'date' in df.columns and row_count > 0:
                first_date = df['date'].iloc[0]
                last_date = df['date'].iloc[-1]
                
                # Lấy tỉ giá
                if 'rate' in df.columns:
                    first_rate = df['rate'].iloc[0]
                    last_rate = df['rate'].iloc[-1]
                    avg_rate = df['rate'].mean()
                    std_rate = df['rate'].std()
                    
                    # Tính thay đổi phần trăm
                    if first_rate != 0:
                        pct_change = ((last_rate - first_rate) / first_rate) * 100
                    else:
                        pct_change = 0
                else:
                    first_rate = last_rate = avg_rate = std_rate = pct_change = 'N/A'
                
                stats_data.append({
                    'currency_code': file_name.upper(),
                    'currency_name': file_name.replace('_', ' ').title(),
                    'data_points': row_count,
                    'first_date': first_date,
                    'last_date': last_date,
                    'days_covered': (pd.to_datetime(last_date) - pd.to_datetime(first_date)).days,
                    'first_rate': first_rate if first_rate != 'N/A' else None,
                    'last_rate': last_rate if last_rate != 'N/A' else None,
                    'avg_rate': avg_rate if avg_rate != 'N/A' else None,
                    'std_rate': std_rate if std_rate != 'N/A' else None,
                    'pct_change': pct_change if pct_change != 'N/A' else None
                })
                
                print(f"✓ {file_name:25s} - {row_count:4d} dòng ({first_date} → {last_date})")
            else:
                print(f"⚠️  {file_name:25s} - Cấu trúc file không đúng")
                
        except Exception as e:
            print(f"✗ {file.stem:25s} - Lỗi: {e}")
    
    if stats_data:
        # Tạo DataFrame và lưu
        stats_df = pd.DataFrame(stats_data)
        stats_file = currency_dir / "detailed_statistics.csv"
        stats_df.to_csv(stats_file, index=False, encoding='utf-8-sig')
        
        print(f"\n✅ Đã lưu thống kê chi tiết: {stats_file}")
        
        # Hiển thị summary
        print("\n📊 TỔNG KẾT THỐNG KÊ:")
        print(f"   • Tổng đồng tiền: {len(stats_df)}")
        print(f"   • Tổng số dòng dữ liệu: {stats_df['data_points'].sum():,}")
        print(f"   • Phạm vi ngày trung bình: {stats_df['days_covered'].mean():.0f} ngày")
        print(f"   • Đồng tiền nhiều dữ liệu nhất: {stats_df.loc[stats_df['data_points'].idxmax(), 'currency_name']} ({stats_df['data_points'].max()} dòng)")
        print(f"   • Đồng tiền ít dữ liệu nhất: {stats_df.loc[stats_df['data_points'].idxmin(), 'currency_name']} ({stats_df['data_points'].min()} dòng)")
        
        # Top 10 đồng tiền có dữ liệu nhiều nhất
        print("\n🏆 TOP 10 ĐỒNG TIỀN CÓ NHIỀU DỮ LIỆU NHẤT:")
        top10 = stats_df.nlargest(10, 'data_points')[['currency_name', 'data_points', 'first_date', 'last_date']]
        for i, (_, row) in enumerate(top10.iterrows(), 1):
            print(f"   {i:2d}. {row['currency_name']:25s} - {row['data_points']:5d} dòng")

def analyze_currency_structure():
    """Phân tích cấu trúc dữ liệu của các file"""
    print("\n" + "=" * 80)
    print("🔍 PHÂN TÍCH CẤU TRÚC DỮ LIỆU")
    print("=" * 80)
    
    currency_dir = Path("D:/FPT/ki 7/DAT/project_exchange_rate/exchange-rate-forecast/data/processed/currencies")
    currency_files = list(currency_dir.glob("*.csv"))
    currency_files = [f for f in currency_files if "summary" not in f.name.lower() and "statistics" not in f.name.lower()]
    
    print("Kiểm tra cấu trúc 5 file đầu tiên:")
    
    for i, file in enumerate(currency_files[:5], 1):
        try:
            df = pd.read_csv(file, nrows=3)
            print(f"\n{i}. {file.name}:")
            print(f"   • Columns: {list(df.columns)}")
            print(f"   • Shape: {df.shape}")
            print(f"   • Sample data:")
            for _, row in df.iterrows():
                if 'date' in df.columns and 'rate' in df.columns:
                    print(f"     {row['date']}: {row['rate']}")
                else:
                    print(f"     {row.to_dict()}")
        except Exception as e:
            print(f"\n{i}. {file.name}: Lỗi - {e}")

if __name__ == "__main__":
    print("=" * 80)
    print("💰 DANH SÁCH TẤT CẢ ĐỒNG TIỀN TRONG DATASET")
    print("=" * 80)
    
    list_all_currencies()
    analyze_currency_structure()
    
    print("\n" + "=" * 80)
    print("🎉 HOÀN THÀNH LIỆT KÊ")
    print("=" * 80)