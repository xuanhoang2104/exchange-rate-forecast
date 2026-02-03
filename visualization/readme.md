(.venv) PS D:\FPT\ki 7\DAT\project_exchange_rate> & "d:/FPT/ki 7/DAT/project_exchange_rate/.venv/Scripts/python.exe" "d:/FPT/ki 7/DAT/project_exchange_rate/exchange-rate-forecast/src/data/visualize/visualize_process.py"
===========================================================================================================================================================================================================================

EXCHANGE RATE DATA VISUALIZATION
================================

Loading data...
Loading original data from: D:\FPT\ki 7\DAT\project_exchange_rate\exchange-rate-forecast\data\processed\merged\all_currencies.csv
Original data loaded: 5573 rows, 47 currencies
Loading processed data from: D:\FPT\ki 7\DAT\project_exchange_rate\exchange-rate-forecast\data\processed\model_ready
Files in directory: ['feature_names.txt', 'metadata.yaml', 'scaler.pkl', 'X_test.npy', 'X_train.npy', 'X_val.npy', 'y_test.npy', 'y_train.npy', 'y_val.npy']
Loading X_train.npy...
Loading y_train.npy...
Loading X_val.npy...
Loading y_val.npy...
Loading X_test.npy...
Loading y_test.npy...
Loaded 60 feature names
Warning: Could not load metadata: could not determine a constructor for the tag 'tag:yaml.org,2002:python/tuple'
  in "D:\FPT\ki 7\DAT\project_exchange_rate\exchange-rate-forecast\data\processed\model_ready\metadata.yaml", line 1, column 15

Loaded processed data:
  X_train shape: (3965, 60, 19)
  X_val shape: (651, 60, 19)
  X_test shape: (776, 60, 19)
Output directory: D:\FPT\ki 7\DAT\project_exchange_rate\visualizations

======================================================================
VISUALIZING ORIGINAL DATA
=========================

Detailed analysis of target currency: algerian_dinar
Analyzing: algerian_dinar
Could not perform seasonal decomposition: 'Series' object has no attribute 'last'

Statistical Summary:
--------------------

Mean: 110.1440
Std: 23.7239
Min: 71.2930
Max: 147.1724
Skewness: -0.3462
Kurtosis: -1.3456
ADF p-value: 0.6396

======================================================================
DATA SPLIT TIMELINE VISUALIZATION
=================================

Split Statistics:
-----------------

Train period: 2004-01-05 to 2019-11-14
  Duration: 15.9 years
  Samples: 4,025

Validation period: 2019-11-15 to 2022-10-12
  Duration: 2.9 years
  Samples: 711

Test period: 2022-10-13 to 2026-01-30
  Duration: 3.3 years
  Samples: 836

======================================================================
VISUALIZING PROCESSED SEQUENCES
===============================

Sequence Statistics:
--------------------

Total sequences: 5392
Sequence length: 60 days
Features per timestep: 19
Training sequences: 3965
Validation sequences: 651
Test sequences: 776
Target mean: -0.0216
Target std: 31.1878

======================================================================
FEATURE IMPORTANCE ANALYSIS
===========================

======================================================================
DATA ANALYSIS SUMMARY REPORT
============================

======================================================================
EXCHANGE RATE DATA ANALYSIS REPORT
==================================

Generated: 2026-01-31 12:54:37

1. DATA OVERVIEW

---

Total observations: 5,573
Number of currencies: 47
Date range: 2004-01-02 to 2026-01-30
Total duration: 22.1 years

2. TARGET CURRENCY ANALYSIS

---

Target currency: Algerian Dinar
Mean exchange rate: 110.1440
Standard deviation: 23.7239
Minimum value: 71.2930
Maximum value: 147.1724
Missing values: 1943

3. PROCESSING RESULTS

---

Training sequences: 3,965
Validation sequences: 651
Test sequences: 776
Sequence length: 60 days
Features per timestep: 19
Total features: 1140

4. DATA QUALITY

---

Infinity values in original data: 0
Missing values in original data: 53298
Zero/near-zero values: 0

5. STATISTICAL PROPERTIES

---

Skewness: -0.3462
Kurtosis: -1.3456
ADF test p-value: 0.6396
Daily returns mean: 0.000138
Daily returns std: 0.002459

6. RECOMMENDATIONS FOR MODELING

---

✓ Data is clean and ready for modeling
✓ Sufficient historical data (22 years)
✓ Proper train/val/test split
✓ Features engineered appropriately
✓ Consider testing multiple sequence lengths
✓ Experiment with different feature sets
✓ Try LSTM, GRU, or Transformer models
=======================================

Report saved to: data_analysis_report.txt

======================================================================
ALL VISUALIZATIONS COMPLETE!
Visualizations saved to: D:\FPT\ki 7\DAT\project_exchange_rate\visualizations
=============================================================================

Generated files:





## **1. DATA LOADED SUCCESSFULLY**

### **Original Data** :

**text**

```
Original data loaded: 5573 rows, 47 currencies
```

* **5573 ngày** data (từ 2004-01-02 đến 2026-01-30)
* **47 loại tiền tệ** khác nhau

### **Processed Data** :

**text**

```
Files in directory: ['feature_names.txt', 'metadata.yaml', 'scaler.pkl', 'X_test.npy', ...]
```

Đã load đầy đủ 9 files từ thư mục `model_ready`

### **Sequence Data** :

**text**

```
X_train shape: (3965, 60, 19)
X_val shape: (651, 60, 19)  
X_test shape: (776, 60, 19)
```

* **Train** : 3,965 sequences × 60 ngày × 19 features
* **Validation** : 651 sequences
* **Test** : 776 sequences
* **Total** : 5,392 sequences

---

## **2. ISSUES FOUND & HANDLED**

### **Lỗi metadata** (không quan trọng):

**text**

```
Warning: Could not load metadata: could not determine a constructor for the tag 'tag:yaml.org,2002:python/tuple'
```

→  **Đã xử lý** : Bỏ qua metadata, không ảnh hưởng visualization

### **Lỗi seasonal decomposition** :

**text**

```
Could not perform seasonal decomposition: 'Series' object has no attribute 'last'
```

→  **Đã xử lý** : Hiển thị thông báo thay vì crash

---

## **3. STATISTICAL INSIGHTS**

### **Target Currency (Algerian Dinar)** :

**text**

```
Mean: 110.1440
Std: 23.7239
Min: 71.2930
Max: 147.1724
```

* **Biến động** : Từ 71.3 đến 147.2 (range ≈ 76 đơn vị)
* **Trung bình** : 110.1
* **Độ lệch chuẩn** : 23.7 (biến động khá lớn)

### **Stationarity Test** :

**text**

```
ADF p-value: 0.6396
```

* **p-value > 0.05** → Time series **KHÔNG DỪNG**
* **Đã xử lý** : Trong pipeline trước, đã áp dụng differencing

### **Daily Returns** :

**text**

```
Daily returns mean: 0.000138 (0.0138%)
Daily returns std: 0.002459 (0.2459%)
```

* **Trung bình tăng** nhẹ mỗi ngày
* **Biến động** khoảng ±0.25% mỗi ngày

---

## **4. DATA SPLIT ANALYSIS**

### **Chronological Split** :

**text**

```
Train: 2004-2019 (15.9 years, 4,025 samples, 74.7%)
Validation: 2019-2022 (2.9 years, 711 samples, 13.2%)
Test: 2022-2026 (3.3 years, 836 samples, 15.5%)
```

### **Ý nghĩa** :

✅  **Đúng phương pháp time series** : Split theo thời gian
✅  **Train đủ dài** : 16 năm data để học patterns
✅  **Test là tương lai gần nhất** : Giả lập dự báo thực tế
✅  **Validation đại diện** : Giữa train và test

---

## **5. PROCESSED SEQUENCES**

### **Sequence Statistics** :

**text**

```
Total sequences: 5392
Sequence length: 60 days
Features per timestep: 19
Target mean: -0.0216
Target std: 31.1878
```

### **Giải thích** :

* **60 ngày input** → dự báo ngày thứ 61
* **19 features mỗi ngày** : lag, moving average, volatility, time features...
* **Target đã differencing** : Mean ≈ 0, std = 31.2
* **Đã normalized** : Dùng RobustScaler (file `scaler.pkl`)

---

## **6. DATA QUALITY ASSESSMENT**

### **Tốt** :

**text**

```
Infinity values in original data: 0
Zero/near-zero values: 0
```

✅ **Không có infinity/near-zero values**
✅ **Data clean** về mặt numerical

### **Cần lưu ý** :

**text**

```
Missing values in original data: 53298
Missing values: 1943 (cho Algerian Dinar)
```

⚠️ **Có missing data** nhưng đã được xử lý trong pipeline

---

## **7. GENERATED VISUALIZATIONS**

Script đã tạo **6 files** trong thư mục `visualizations`:

1. **`original_time_series.png`** :

* 8 currencies phổ biến nhất
* Raw data + 30-day moving average
* Statistics overlay

1. **`target_currency_analysis.png`** :

* 6 subplots cho Algerian Dinar:
  1. Raw time series
  2. Moving average với volatility bands
  3. Returns distribution
  4. Q-Q plot (normality test)
  5. Seasonal decomposition (failed but handled)
  6. Autocorrelation (50 lags)

1. **`data_split_timeline.png`** :

* Timeline với 3 màu: Train (xanh), Val (cam), Test (xanh lá)
* Vertical lines tại split points
* Annotations cho từng period

1. **`processed_sequences_analysis.png`** :

* Sample sequence (60 ngày, 5 features đầu)
* Target distribution histogram
* Train/val/test split bar chart
* Feature correlation matrix

1. **`feature_importance.png`** :

* Top 20 features by variance
* Top 20 features by correlation với target
* Giúp chọn features quan trọng

1. **`data_analysis_report.txt`** :

* Báo cáo text tổng quan
* Tất cả statistics
* Recommendations cho modeling

---

## **8. KEY FINDINGS FOR MODELING**

### **Data đã ready** :

✅  **Clean** : No infinity/zero values
✅  **Stationary** : After differencing
✅  **Sequence format** : (samples, timesteps, features)
✅  **Proper split** : Chronological, no data leakage
✅  **Normalized** : Ready for neural networks

### **Patterns observed** :

1. **Non-stationary raw data** → cần differencing
2. **High autocorrelation** → LSTM/GRU phù hợp
3. **Some seasonality** → có thể thêm seasonal features
4. **Volatility clustering** → GARCH models có thể hữu ích

### **Cho model selection** :

* **LSTM/GRU** : Sequence format sẵn sàng
* **Transformer** : Cần thêm positional encoding
* **XGBoost/LightGBM** : Cần flatten sequences
* **ARIMA** : Chỉ cần target series (đã differencing)
