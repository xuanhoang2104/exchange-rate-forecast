#!/usr/bin/env python3
"""
Lay du lieu USD/VND tu Free Currency API va luu vao file vnd.csv
Tu 2020-01-30 den nay
"""

import pandas as pd
import requests
from datetime import datetime, timedelta
import time
import os
from pathlib import Path
import sys

def print_progress(processed, total, current_date, success_count, failed_count):
    """Hien thi progress bar"""
    bar_length = 50
    percent = float(processed) / total
    arrow = '=' * int(round(percent * bar_length) - 1) + '>'
    spaces = ' ' * (bar_length - len(arrow))
    
    sys.stdout.write(f"\r  [")
    sys.stdout.write(f"{arrow}{spaces}")
    sys.stdout.write(f"] {int(round(percent * 100))}% ")
    sys.stdout.write(f"({processed}/{total}) {current_date.strftime('%Y-%m-%d')} ")
    sys.stdout.write(f"✅:{success_count} ❌:{failed_count}")
    sys.stdout.flush()

def fetch_vnd_data_single_day(date_str, retry_count=2):
    """
    Lay du lieu cho mot ngay cu the
    """
    url = f"https://cdn.jsdelivr.net/npm/@fawazahmed0/currency-api@{date_str}/v1/currencies/usd.json"
    
    for attempt in range(retry_count):
        try:
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                vnd_rate = data.get('usd', {}).get('vnd')
                
                if vnd_rate:
                    return {
                        'date': date_str,
                        'rate': float(vnd_rate),
                        'status': 'success'
                    }
                else:
                    return {
                        'date': date_str,
                        'rate': None,
                        'status': 'no_data'
                    }
            elif response.status_code == 404:
                return {
                    'date': date_str,
                    'rate': None,
                    'status': 'not_found'
                }
            else:
                if attempt < retry_count - 1:
                    time.sleep(1)
                    continue
                return {
                    'date': date_str,
                    'rate': None,
                    'status': f'http_{response.status_code}'
                }
                
        except requests.exceptions.Timeout:
            if attempt < retry_count - 1:
                time.sleep(2)
                continue
            return {
                'date': date_str,
                'rate': None,
                'status': 'timeout'
            }
        except Exception as e:
            if attempt < retry_count - 1:
                time.sleep(1)
                continue
            return {
                'date': date_str,
                'rate': None,
                'status': f'error_{type(e).__name__}'
            }
    
    return {
        'date': date_str,
        'rate': None,
        'status': 'failed'
    }

def fetch_vnd_data_range(start_date="2020-01-30", end_date=None):
    """
    Lay du lieu USD/VND tu mot khoang thoi gian
    """
    print("\n" + "="*70)
    print("LAY DU LIEU USD/VND")
    print("="*70)
    
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    
    print(f"Thoi gian: {start_date} -> {end_date}")
    
    # Tao danh sach tat ca cac ngay
    all_dates = []
    current_date = start
    while current_date <= end:
        # Chi lay ngay lam viec (thu 2-6)
        if current_date.weekday() < 5:
            all_dates.append(current_date.strftime("%Y-%m-%d"))
        current_date += timedelta(days=1)
    
    total_days = len(all_dates)
    print(f"Tong ngay lam viec can lay: {total_days} ngay")
    print("Dang bat dau lay du lieu...")
    print()
    
    results = []
    processed = 0
    success_count = 0
    failed_count = 0
    
    for date_str in all_dates:
        current_date = datetime.strptime(date_str, "%Y-%m-%d")
        
        result = fetch_vnd_data_single_day(date_str)
        results.append(result)
        
        if result['status'] == 'success':
            success_count += 1
        else:
            failed_count += 1
        
        processed += 1
        
        # Cap nhat progress bar moi 10 ngay hoac khi hoan thanh
        if processed % 10 == 0 or processed == total_days:
            print_progress(processed, total_days, current_date, success_count, failed_count)
        
        time.sleep(0.1)  # Delay de khong spam server
    
    print("\n\n" + "="*70)
    print("KET QUA LAY DU LIEU")
    print("="*70)
    
    # Phan tich ket qua
    success_data = [r for r in results if r['status'] == 'success']
    not_found_data = [r for r in results if r['status'] == 'not_found']
    no_data = [r for r in results if r['status'] == 'no_data']
    other_errors = [r for r in results if r['status'] not in ['success', 'not_found', 'no_data']]
    
    print(f"Thanh cong: {len(success_data)}/{total_days} ngay ({len(success_data)/total_days*100:.1f}%)")
    print(f"Khong tim thay (404): {len(not_found_data)} ngay")
    print(f"Khong co du lieu VND: {len(no_data)} ngay")
    print(f"Loi khac: {len(other_errors)} ngay")
    
    # Thong ke theo nam
    if success_data:
        print("\n" + "-"*40)
        print("THONG KE THEO NAM:")
        print("-"*40)
        
        years_data = {}
        for item in success_data:
            year = item['date'][:4]
            if year not in years_data:
                years_data[year] = []
            years_data[year].append(item)
        
        for year in sorted(years_data.keys()):
            count = len(years_data[year])
            if count > 0:
                rates = [item['rate'] for item in years_data[year]]
                avg_rate = sum(rates) / len(rates)
                print(f"{year}: {count} ngay | TB: {avg_rate:,.0f} VND/USD")
    
    if success_data:
        # Tao DataFrame tu du lieu thanh cong
        data_list = []
        for item in success_data:
            data_list.append({
                'date': item['date'],
                'rate': item['rate']
            })
        
        df = pd.DataFrame(data_list)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        print(f"\nDu lieu co san tu: {df['date'].min().date()} den {df['date'].max().date()}")
        
        # Kiem tra khoang trong du lieu lon (> 10 ngay)
        if len(df) > 1:
            date_diffs = df['date'].diff().dt.days
            big_gaps = date_diffs[date_diffs > 10]
            if len(big_gaps) > 0:
                print(f"\nCanh bao: Co {len(big_gaps)} khoang trong du lieu lon (>10 ngay)")
        
        return df
    else:
        print("KHONG LAY DUOC DU LIEU NAO!")
        return None

def check_and_update_existing_data(new_df, output_path):
    """
    Kiem tra va cap nhat du lieu moi vao file cu
    """
    if not os.path.exists(output_path):
        print("File cu khong ton tai. Tao file moi.")
        return new_df
    
    try:
        existing_df = pd.read_csv(output_path)
        existing_df['date'] = pd.to_datetime(existing_df['date'])
        
        print(f"\nFile cu da co {len(existing_df)} dong du lieu")
        print(f"Tu ngay: {existing_df['date'].min().date()} den {existing_df['date'].max().date()}")
        
        # Gop du lieu moi vao du lieu cu
        combined_df = pd.concat([existing_df, new_df]).drop_duplicates(subset=['date']).sort_values('date')
        
        added_count = len(combined_df) - len(existing_df)
        print(f"\nSau khi ket hop:")
        print(f"- Tong so dong: {len(combined_df)}")
        print(f"- Them moi: {added_count} dong")
        print(f"- Tu ngay: {combined_df['date'].min().date()} den {combined_df['date'].max().date()}")
        
        return combined_df
        
    except Exception as e:
        print(f"Loi khi doc file cu: {e}")
        print("Tao file moi...")
        return new_df

def save_vnd_file(df, output_path):
    """Luu file vnd.csv"""
    if df is not None and len(df) > 0:
        # Tao thu muc neu chua ton tai
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Luu file
        df.to_csv(output_path, index=False)
        
        print("\n" + "="*70)
        print("DA LUU FILE THANH CONG")
        print("="*70)
        
        print(f"File: {output_path}")
        print(f"So dong du lieu: {len(df)}")
        print(f"Pham vi: {df['date'].min().date()} -> {df['date'].max().date()}")
        
        if len(df) > 0:
            print(f"\nThong ke ty gia:")
            print(f"- Trung binh: {df['rate'].mean():,.2f} VND/USD")
            print(f"- Cao nhat: {df['rate'].max():,.2f} VND/USD (ngay: {df.loc[df['rate'].idxmax(), 'date'].date()})")
            print(f"- Thap nhat: {df['rate'].min():,.2f} VND/USD (ngay: {df.loc[df['rate'].idxmin(), 'date'].date()})")
            
            # Tinh % thay doi
            if len(df) > 1:
                first_rate = df['rate'].iloc[0]
                last_rate = df['rate'].iloc[-1]
                pct_change = ((last_rate - first_rate) / first_rate) * 100
                print(f"- % thay doi tu dau: {pct_change:+.2f}%")
            
            print(f"\n5 gia tri moi nhat:")
            for i in range(min(5, len(df))):
                idx = len(df) - 1 - i
                print(f"  {df.iloc[idx]['date'].date()}: {df.iloc[idx]['rate']:,.2f}")
        
        return True
    else:
        print("Khong co du lieu de luu!")
        return False

def main():
    """Ham chinh"""
    print("BAT DAU CHUONG TRINH LAY DU LIEU VND")
    print("Duong dan luu: D:/FPT/ki 7/DAT/project_exchange_rate/exchange-rate-forecast/data/processed/currencies/vnd.csv")
    
    # Duong dan luu file
    output_path = r"D:\FPT\ki 7\DAT\project_exchange_rate\exchange-rate-forecast\data\processed\currencies\vnd.csv"
    
    # Hoi nguoi dung
    print("\n" + "="*70)
    print("TUY CHON LAY DU LIEU")
    print("="*70)
    print("1. Lay tu dau (2020-01-30 -> hien tai)")
    print("2. Chi lay du lieu moi (tu ngay cuoi cung trong file)")
    print("3. Huy")
    
    choice = input("\nChon tuy chon [1/2/3]: ")
    
    if choice == '3':
        print("Da huy!")
        return
    
    start_time = time.time()
    
    try:
        if choice == '2' and os.path.exists(output_path):
            try:
                # Doc ngay cuoi cung tu file
                existing_df = pd.read_csv(output_path)
                existing_df['date'] = pd.to_datetime(existing_df['date'])
                last_date = existing_df['date'].max()
                start_date = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
                print(f"\nSe lay du lieu tu ngay: {start_date}")
            except:
                start_date = "2020-01-30"
                print(f"\nKhong doc duoc file cu. Bat dau tu: {start_date}")
        else:
            start_date = "2020-01-30"
            print(f"\nBat dau lay du lieu tu: {start_date}")
        
        # Lay du lieu
        vnd_df = fetch_vnd_data_range(start_date=start_date)
        
        if vnd_df is not None:
            # Kiem tra va cap nhat du lieu cu
            if os.path.exists(output_path) and choice != '1':
                vnd_df = check_and_update_existing_data(vnd_df, output_path)
            
            # Luu file
            success = save_vnd_file(vnd_df, output_path)
            
            if success:
                elapsed = time.time() - start_time
                print(f"\nHOAN THANH SAU {elapsed:.1f} GIAY!")
                print("Du lieu VND da duoc luu thanh cong!")
                
                # Hien thi thong tin file
                if os.path.exists(output_path):
                    file_size = os.path.getsize(output_path) / 1024
                    print(f"Kich thuoc file: {file_size:.1f} KB")
        else:
            print("Khong the lay du lieu!")
            
    except KeyboardInterrupt:
        print("\nDa dung boi nguoi dung!")
    except Exception as e:
        print(f"\nLOI: {type(e).__name__}: {e}")

if __name__ == "__main__":
    main()

"""
Lấy dữ liệu từ 2020-01-30 đến hiện tại

Có progress bar rõ ràng hiển thị:

Phần trăm hoàn thành

Ngày hiện tại đang xử lý

Số ngày thành công/thất bại

Thống kê chi tiết:

Thống kê theo năm

Tỷ giá trung bình, cao nhất, thấp nhất

% thay đổi từ đầu đến cuối

Tùy chọn linh hoạt:

Option 1: Lấy từ đầu (2020-01-30)

Option 2: Chỉ lấy dữ liệu mới từ ngày cuối trong file

Option 3: Hủy

Kiểm tra và cập nhật file cũ nếu đã tồn tại
"""
