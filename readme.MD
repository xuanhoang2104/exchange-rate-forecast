USD Index Forecasting with Multivariate Time Series
1. Giới thiệu

Dự án này tập trung vào việc phân tích và dự báo sức mạnh của đồng đô la Mỹ (USD) dựa trên dữ liệu tỷ giá hối đoái của nhiều đồng tiền lớn trên thế giới.

Thay vì dự báo trực tiếp tỷ giá USD (vốn là hằng số vì USD được dùng làm base currency), dự án xây dựng một chỉ số USD tổng hợp (USD Index) để biểu diễn sức mạnh tương đối của USD so với các đồng tiền chủ chốt.

2. Định nghĩa bài toán
Vấn đề

Trong dataset gốc:

us_dollar_to_usd = 1.0 (không đổi)


→ Không thể sử dụng trực tiếp làm target.

Giải pháp

Xây dựng Synthetic USD Index:
![alt text](image-1.png)


Với N = 7 đồng tiền lớn:

EUR (Euro)

JPY (Japanese Yen)

GBP (UK Pound)

CHF (Swiss Franc)

AUD (Australian Dollar)

CAD (Canadian Dollar)

CNY (Chinese Yuan)

3. Ý nghĩa thực tế

Target usd_index đại diện cho:

Sức mạnh tổng hợp của USD trong hệ thống tài chính toàn cầu.

Ứng dụng:

Nhà đầu tư: xác định xu hướng nắm giữ USD

Doanh nghiệp xuất nhập khẩu: phòng ngừa rủi ro tỷ giá

Ngân hàng & Forex: hedging và risk management

Phân tích vĩ mô: đánh giá chính sách tiền tệ

4. Cấu trúc project
exchange-rate-forecast/
│
├── data/
│   ├── raw/
│   ├── processed/usd/
│
├── scripts/
│   ├── preprocessing/
│   │   ├── build_usd_index.py
│   │   ├── clean_data.py
│   │   ├── stationarity.py
│   │   └── feature_engineering.py
│   │
│   ├── models/
│   │   ├── arima.py
│   │   ├── lstm.py
│   │   └── transformer.py
│   │
│   ├── training/train_model.py
│   └── evaluation/evaluate.py
│
├── config/config.yaml
├── main.py
└── README.md

5. Pipeline xử lý dữ liệu
Raw exchange_rate_to_usd.csv
   ↓
Select major currencies
   ↓
Build USD Index
   ↓
Missing value handling
   ↓
Stationarity (ADF, differencing)
   ↓
Feature engineering (lag, MA, volatility)
   ↓
Scaling & sequence
   ↓
Model training
   ↓
Evaluation

6. Dataset

Nguồn: Historical exchange rates (2004 – nay)

Mỗi cột:

currency_to_usd = 1 đơn vị currency = ? USD


Ví dụ:

euro_to_usd = 1.26 → 1 EUR = 1.26 USD

japanese_yen_to_usd = 0.009 → 1 JPY = 0.009 USD

7. Models

Các mô hình được sử dụng:

Model	Mục đích
ARIMA	Baseline thống kê
LSTM	Deep learning
Transformer	Mô hình hiện đại
8. Evaluation

Metrics:

MAE (Mean Absolute Error)

RMSE (Root Mean Squared Error)

So sánh hiệu quả giữa các mô hình.

9. Kết luận

Dự án không chỉ minh họa kỹ thuật học máy trên chuỗi thời gian, mà còn giải quyết một bài toán có ý nghĩa kinh tế thực tế:
dự báo sức mạnh của đồng USD trong tương lai.

10. Câu chốt học thuật (để thuyết trình)

“Do USD là base currency nên tỷ giá USD là hằng số. Vì vậy dự án chuyển bài toán sang mô hình hóa sức mạnh tương đối của USD thông qua chỉ số tổng hợp từ các đồng tiền chủ chốt.”


----------------------------------------------------|
Horizon: 7 ngày                                     |
Input: 1 chuỗi USD (univariate)                     |
Pipeline: Baseline → ARIMA → LSTM                   |
----------------------------------------------------|


ví dụ :--------------------------

2024
| Currency | rate_to_usd |
| -------- | ----------- |
| EUR      | 1.10        |
| JPY      | 0.009       |
| GBP      | 1.30        |
| CHF      | 1.12        |
| AUD      | 0.67        |

usd_index = mean(major)
(1.10+0.009+1.30+1.12+0.67)/5=0.9018

2025
| Currency | rate_to_usd |
| -------- | ----------- |
| EUR      | 1.20        |
| JPY      | 0.008       |
| GBP      | 1.40        |
| CHF      | 1.22        |
| AUD      | 0.77        |

usd_index = mean(major)
(1.20+0.008+1.40+1.22+0.77)/5=1.
-> usd_index tăng -> usd mạnh lên