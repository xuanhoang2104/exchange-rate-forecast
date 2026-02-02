KẾT LUẬN – ARIMA BASELINE
1. Mục tiêu

Mục tiêu của giai đoạn này là xây dựng một baseline truyền thống cho bài toán dự đoán sức mạnh đồng USD trong ngắn hạn (7 ngày), dựa trên dữ liệu chuỗi thời gian lịch sử.

2. Tiền xử lý dữ liệu
2.1 Xây dựng USD Index

Thay vì sử dụng trực tiếp chỉ số DXY bên ngoài, nhóm tự xây dựng USD Index nội bộ bằng cách lấy trung bình của các đồng tiền major:

EUR

JPY

GBP

CHF

AUD

CAD

CNY

Điều này giúp:

Kiểm soát được nguồn dữ liệu

Hiểu rõ cấu trúc feature

Phù hợp cho mở rộng sang multivariate sau này

2.2 Chuẩn hóa chuỗi

Chuỗi USD Index ban đầu không dừng (non-stationary), được xác nhận bằng:

ADF test: p-value = 0.0018
→ Bác bỏ H0 → chuỗi có xu hướng.

Áp dụng:

log transform + differencing


Sau khi xử lý:

ADF sau diff: p-value = 0.0
→ Chuỗi đã stationarity.

3. Huấn luyện ARIMA
3.1 Tìm tham số tối ưu

Sử dụng auto_arima để tìm bộ (p,d,q) tối ưu dựa trên:

AIC

BIC

Kết quả:

Best model: ARIMA(0,0,2)
AIC = -2595

3.2 Đánh giá hiệu năng

Sử dụng:

Hold-out test (80/20)

Rolling backtest

Kết quả:

MAE ≈ 0.052


Đây là mức sai số chấp nhận được cho baseline.

4. Hiện tượng "đường phẳng"

Khi dự đoán nhiều bước (14–28 ngày), mô hình cho kết quả:

0.741
0.741
0.741
0.741
...

Nguyên nhân toán học:

ARIMA là mô hình tuyến tính, hồi quy nội bộ:

Dự đoán tương lai = tổ hợp tuyến tính của quá khứ + nhiễu trắng.

Khi dự đoán xa:

Không còn thông tin mới

Mô hình hội tụ về kỳ vọng toán học (mean)

→ Dẫn tới đường phẳng.

5. Kết luận khoa học về ARIMA

ARIMA không sai, mà:

Điểm mạnh	Giới hạn
Tốt cho ngắn hạn	Không học được phi tuyến
Dễ giải thích	Không học được regime change
Chuẩn baseline	Không dự báo dài hạn
6. Ý nghĩa của kết quả phẳng

Kết quả này cho thấy:

Cấu trúc dữ liệu USD chứa quan hệ phi tuyến mà mô hình tuyến tính không thể học được.

Nói cách khác:

ARIMA đã “hút hết phần tuyến tính”, phần còn lại là nonlinear structure.

CHUYỂN SANG LSTM
Lý do khoa học

LSTM được chọn vì:

Có memory cell

Học được:

Pattern dài hạn

Quan hệ phi tuyến

Regime shift

Khác với ARIMA:

ARIMA	LSTM
Linear	Nonlinear
Không nhớ dài hạn	Có long-term memory
Hội tụ mean	Có thể tạo trajectory
CHUYỂN SANG TRANSFORMER
Lý do khoa học

Transformer được chọn để:

Attention toàn chuỗi

Không bị giới hạn cửa sổ 30 ngày

Học:

Quan hệ giữa các currency

Shock toàn cục

Lead-lag effects

KẾT LUẬN TỔNG QUÁT

ARIMA được sử dụng như một baseline khoa học, giúp:

Chuẩn hóa pipeline

Xác nhận stationarity

Đánh giá độ khó của bài toán

Tuy nhiên:

Do bản chất tuyến tính, ARIMA không thể mô hình hóa đầy đủ động lực thị trường ngoại hối.

****
KẾT LUẬN KHOA HỌC (RẤT QUAN TRỌNG)

Bạn đã chứng minh được 3 điều:

✅ 1. ARIMA fit dữ liệu tốt

AIC rất thấp

MAE nhỏ

Residual white-noise

❌ 2. ARIMA dự báo dài hạn → phẳng

Mean reversion

Không học được nonlinear

Không phản ứng shock

✅ 3. Backtest cho thấy hạn chế thực tế

Rolling std cao

Không ổn định khi regime change


----------------------------------------------------------------------------------------------
Bài toán

Dự đoán sức mạnh đồng USD trong tương lai 7 ngày dựa trên dữ liệu lịch sử các tỷ giá major.

GIAI ĐOẠN 1 – HIỂU VÀ CHUẨN HÓA DỮ LIỆU
Bước 1: Thu thập dữ liệu

Dữ liệu gồm:

Tỷ giá nhiều đồng tiền so với USD từ 2004–2026.

Mỗi dòng:

date, euro_to_usd, jpy_to_usd, ...

Bước 2: Xây dựng USD Index

Thay vì dùng DXY bên ngoài:

Tạo chỉ số nội bộ:

USD_index(t) = mean(EUR, JPY, GBP, CHF, AUD, CAD, CNY)


→ Biến bài toán từ multivariate → univariate.

Bước 3: Phân tích chuỗi thời gian (EDA)

Thực hiện:

Plot raw series

Rolling volatility

Histogram

Log transform

Phát hiện:

Chuỗi có trend

Variance không ổn định

Non-stationary

Bước 4: Kiểm tra tính dừng (Stationarity)

Dùng ADF test:

ADF p-value > 0.05 → không dừng

Sau log + diff:

ADF p-value ≈ 0 → dừng

→ Thỏa điều kiện cho ARIMA.

GIAI ĐOẠN 2 – BASELINE VỚI ARIMA
Bước 5: Chọn mô hình baseline

Chọn ARIMA vì:

Chuẩn thống kê

Dễ giải thích

Là mốc so sánh cho deep learning

Bước 6: Tìm tham số tối ưu

Dùng auto_arima:

Tối ưu theo:

AIC

BIC

Kết quả:

ARIMA(0,0,2)

Bước 7: Đánh giá mô hình

Hai cách:

7.1 Hold-out test

80% train – 20% test
→ MAE ≈ 0.052

7.2 Rolling backtest

Mỗi bước:

Train lại

Predict 7 ngày

Tính lỗi

→ Đánh giá thực tế như trading.

Bước 8: Phân tích kết quả

Dự đoán dài hạn → đường phẳng.

Nguyên nhân:

ARIMA là tuyến tính

Không học được nonlinear pattern

Hội tụ mean

KẾT LUẬN GIAI ĐOẠN 1

ARIMA chỉ học được phần tuyến tính của dữ liệu.
Phần động lực chính của thị trường ngoại hối là phi tuyến.

GIAI ĐOẠN 3 – CHUYỂN SANG LSTM
Mục tiêu

Học được:

Quan hệ dài hạn

Phi tuyến

Memory effect

Các bước tiếp theo
Bước 9: Tạo supervised data

Windowing:

30 ngày → dự đoán ngày thứ 38


Dạng:

X: (N, 30, 1)
y: (N,)

Bước 10: Train LSTM

Kiến trúc:

Input → LSTM → Dense → Output


So sánh:

MAE LSTM vs ARIMA

Visual forecast

GIAI ĐOẠN 4 – TRANSFORMER
Mục tiêu

Bắt:

Regime shift

Lead-lag giữa currency

Shock toàn cục

Cách làm
Multivariate input:
(EUR, JPY, GBP, CHF, AUD, CAD, CNY)


Dạng:

X: (N, 30, 7)


Dùng:

Attention

Positional encoding

THỨ TỰ TRIỂN KHAI CHUẨN
Bước	Việc
1	Clean & merge
2	Build USD index
3	EDA
4	Stationarity
5	ARIMA baseline
6	Backtest
7	App ARIMA
8	LSTM
9	Transformer
10	So sánh & kết luận



