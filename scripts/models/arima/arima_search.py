import pandas as pd
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error

series = pd.read_csv("data/processed/model_ready/usd_series.csv",
                     index_col="date")["usd_index"]

split = int(len(series)*0.8)
train, test = series[:split], series[split:]

model = auto_arima(
    train,
    start_p=0, start_q=0,
    max_p=5, max_q=5,
    d=0,     # diff ở understand_data.ipynb
    seasonal=False,
    trace=True,
    error_action="ignore",
    suppress_warnings=True
)

print(model.summary())

pred = model.predict(n_periods=len(test))
mae = mean_absolute_error(test, pred)
print("Best ARIMA MAE:", mae)

"""
AIC thấp nhất:
AIC = Akaike Information Criterion
AIC=2k−2ln(L)

k: số tham số
L: likelihood
-------------------------------


BIC thấp nhất
BIC=kln(n)−2ln(L)

phạt phức tạp mạnh hơn
đặc biệt khi dữ liệu lớn

-------------------------------
ví dụ:
Model	AIC
ARIMA(1,0,1)	-1200
ARIMA(2,0,1)	-1250
ARIMA(5,0,5)	-1300

Thoạt nhìn:

(5,0,5) tốt nhất

Nhưng BIC:

Model	BIC
(1,0,1)	-1190
(2,0,1)	-1240
(5,0,5)	-1100

BIC nói:

(5,0,5) quá phức tạp → loại


Ljung-Box test 
Prob(Q): 0.82


Nghĩa là:

residual không còn autocorrelation

→ model đã hút hết structure trong data
→ đúng chuẩn ARIMA.


Best model:  ARIMA(0,0,2)(0,0,0)[0]
Total fit time: 5.227 seconds
                               SARIMAX Results
==============================================================================
Dep. Variable:                      y   No. Observations:                 4427
Model:               SARIMAX(0, 0, 2)   Log Likelihood                1300.791
Date:                Mon, 02 Feb 2026   AIC                          -2595.582
Time:                        15:37:36   BIC                          -2576.396
Sample:                             0   HQIC                         -2588.816
                               - 4427
Covariance Type:                  opg
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
ma.L1         -0.7668      0.003   -263.653      0.000      -0.772      -0.761
ma.L2         -0.2069      0.005    -38.951      0.000      -0.217      -0.197
sigma2         0.0325   6.79e-05    478.459      0.000       0.032       0.033
===================================================================================
Ljung-Box (L1) (Q):                   0.05   Jarque-Bera (JB):          10781388.11
Prob(Q):                              0.82   Prob(JB):                         0.00
Heteroskedasticity (H):               0.30   Skew:                           -13.27
Prob(H) (two-sided):                  0.00   Kurtosis:                       243.30
===================================================================================

Warnings:
[1] Covariance matrix calculated using the outer product of gradients (complex-step).
D:\FPT\kì 7\DAT\exchange-rate\.venv\Lib\site-packages\statsmodels\tsa\base\tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
D:\FPT\kì 7\DAT\exchange-rate\.venv\Lib\site-packages\statsmodels\tsa\base\tsa_model.py:837: FutureWarning: No supported index is available. In the next version, calling this method in a model without a supported index will result in an exception.
  return get_prediction_index(
Best ARIMA MAE: 0.052797135646146194

"""