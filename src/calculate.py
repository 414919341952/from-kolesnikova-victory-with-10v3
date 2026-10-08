import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import statsmodels.api as sm

from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan
from statsmodels.stats.stattools import jarque_bera
from sklearn.metrics import mean_squared_error, r2_score

with open('models.pkl', 'rb') as f:
    saved = pickle.load(f)

train             = saved['train']
test              = saved['test']
forecast_arima    = saved['forecast_arima']
forecast_sarimax  = saved['forecast_sarimax']
results_arima     = saved['results_arima']
results_sarimax   = saved['results_sarimax']
exog_train        = saved['exog_train']
exog_test         = saved['exog_test']

print(f"Загружено из models.pkl")
print(f"Train: {train.index.min().date()} - {train.index.max().date()} ({len(train)} мес.)")
print(f"Test:  {test.index.min().date()} - {test.index.max().date()} ({len(test)} мес.)")

mse_arima = mean_squared_error(test['SalesAmount'], forecast_arima)
r2_arima  = r2_score(test['SalesAmount'], forecast_arima)

mse_sarimax = mean_squared_error(test['SalesAmount'], forecast_sarimax)
r2_sarimax  = r2_score(test['SalesAmount'], forecast_sarimax)

metrics_df = pd.DataFrame({
    'Модель': ['ARIMA(1,1,1)', 'SARIMAX(1,1,1)(1,1,1,12)'],
    'MSE':    [mse_arima, mse_sarimax],
    'R²':     [r2_arima, r2_sarimax],
    'AIC':    [results_arima.aic, results_sarimax.aic],
    'BIC':    [results_arima.bic, results_sarimax.bic],
})

print("\n=== Таблица метрик ===")
print(metrics_df.round(2).to_string(index=False))

# Сохраним таблицу в CSV для отчета
metrics_df.round(3).to_csv('metrics.csv', index=False, encoding='utf-8-sig')
print("Сохранено: metrics.csv")

def analyze_residuals(results, model_name, exog=None):
    print(f"\n--- Анализ остатков: {model_name} ---")

    # Приводим остатки к массиву и убираем NaN (первые точки после дифференцирования)
    resid = np.asarray(results.resid).flatten()
    resid_clean = resid[~np.isnan(resid)]
    print(f"Наблюдений: всего={len(resid)}, после удаления NaN={len(resid_clean)}")

    _, jb_pvalue, _, _ = jarque_bera(resid_clean)
    verdict_norm = "OK" if jb_pvalue > 0.05 else "НАРУШЕНА"
    print(f"Jarque-Bera p-value: {jb_pvalue:.4f} -> нормальность {verdict_norm}")

    lb = acorr_ljungbox(resid_clean, lags=[12], return_df=True)
    lb_pvalue = lb['lb_pvalue'].values[0]
    verdict_ac = "OK" if lb_pvalue > 0.05 else "ЕСТЬ АВТОКОРРЕЛЯЦИЯ"
    print(f"Ljung-Box (lag=12) p-value: {lb_pvalue:.4f} -> {verdict_ac}")

    if exog is not None:
        exog_arr = np.asarray(exog)
        if exog_arr.ndim == 1:
            exog_arr = exog_arr.reshape(-1, 1)

        n_diff = len(exog_arr) - len(resid_clean)
        exog_aligned = exog_arr[n_diff:] if n_diff > 0 else exog_arr

        exog_bp = sm.add_constant(exog_aligned, has_constant='add')

        try:
            _, bp_pvalue, _, _ = het_breuschpagan(resid_clean, exog_bp)
            verdict_hs = "OK" if bp_pvalue > 0.05 else "ГЕТЕРОСКЕДАСТИЧНОСТЬ"
            print(f"Breusch-Pagan p-value: {bp_pvalue:.4f} -> {verdict_hs}")
        except Exception as e:
            print(f"Breusch-Pagan: не удалось выполнить ({e})")
    else:
        print("Breusch-Pagan: пропущен (нет экзогенных переменных)")

# ARIMA — без экзогенных переменных
analyze_residuals(results_arima, "ARIMA(1,1,1)", exog=None)
# SARIMAX — с экзогенными переменными из train
analyze_residuals(results_sarimax, "SARIMAX(1,1,1)(1,1,1,12)", exog=exog_train)

for res, name in [(results_arima, 'ARIMA'), (results_sarimax, 'SARIMAX')]:
    fig = res.plot_diagnostics(figsize=(14, 8))
    fig.suptitle(f'Диагностика остатков: {name}', y=1.02)
    fig.savefig(f'diagnostics_{name.lower()}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Сохранено: diagnostics_{name.lower()}.png")

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
plt.savefig('forecast_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Сохранено: forecast_comparison.png")
