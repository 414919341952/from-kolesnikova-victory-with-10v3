import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error

df = pd.read_csv('./data/sales.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df = df.asfreq('MS')

train = df.iloc[:-12]
test = df.iloc[-12:]

print(f"Train: {train.index.min()} - {train.index.max()} ({len(train)} мес.)")
print(f"Test:  {test.index.min()} - {test.index.max()} ({len(test)} мес.)")

##################################################################
##################################################################

# --- 1. Модель ARIMA ---
# Используем порядок (p,d,q) = (1,1,1)
model_arima = ARIMA(train['SalesAmount'], order=(1, 1, 1))
results_arima = model_arima.fit()

# Прогноз на 12 месяцев
forecast_arima = results_arima.forecast(steps=len(test))

# Используем порядок (p,d,q) = (1,1,1) и сезонный (P,D,Q,s) = (1,1,1,12)
# Экзогенные переменные: Promotion, HolidayMonth
exog_train = train[['Promotion', 'HolidayMonth']]
exog_test = test[['Promotion', 'HolidayMonth']]

model_sarimax = SARIMAX(train['SalesAmount'], 
                        exog=exog_train,
                        order=(1, 1, 1), 
                        seasonal_order=(1, 1, 1, 12))
results_sarimax = model_sarimax.fit(disp=False)

# Прогноз на 12 месяцев с учетом известных экзогенных переменных на тесте
forecast_sarimax = results_sarimax.forecast(steps=len(test), exog=exog_test)

##################################################################
##################################################################

# Расчет метрик
def evaluate_model(actual, predicted, name):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    print(f"--- {name} ---")
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAPE: {mape:.2f}%\n")
    return mae, rmse, mape

evaluate_model(test['SalesAmount'], forecast_arima, "ARIMA(1,1,1)")
evaluate_model(test['SalesAmount'], forecast_sarimax, "SARIMAX(1,1,1)(1,1,1,12)")

# Визуализация
plt.figure(figsize=(14, 6))
plt.plot(train.index, train['SalesAmount'], label='Train (2020-2022)')
plt.plot(test.index, test['SalesAmount'], label='Test (2023)', color='black', marker='o')
plt.plot(test.index, forecast_arima, label='ARIMA Forecast', color='blue', linestyle='--')
plt.plot(test.index, forecast_sarimax, label='SARIMAX Forecast', color='red', linestyle='--')
plt.title('Прогноз продаж: ARIMA vs SARIMAX')
plt.xlabel('Дата')
plt.ylabel('Сумма продаж')
plt.legend()
plt.grid(True)
plt.savefig('plot-trainer.png', dpi=150, bbox_inches='tight')

import pickle

with open('models.pkl', 'wb') as f:
    pickle.dump({
        'train': train,
        'test': test,
        'forecast_arima': forecast_arima,
        'forecast_sarimax': forecast_sarimax,
        'results_arima': results_arima,
        'results_sarimax': results_sarimax,
        'exog_train': exog_train,
        'exog_test': exog_test,
    }, f)
print("Объекты сохранены в models.pkl")
