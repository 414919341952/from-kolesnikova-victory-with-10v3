import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose, STL
from scipy.fft import fft, fftfreq
import pywt

df = pd.read_csv('./data/sales.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df = df.asfreq('MS')

plt.figure(figsize=(14, 6))
plt.plot(df.index, df['SalesAmount'], marker='o', label='Sales Amount')
plt.title('Динамика продаж (2020-2023)')
plt.xlabel('Дата')
plt.ylabel('Сумма продаж')
plt.grid(True)
plt.legend()
plt.savefig('plot-1.png', dpi=150, bbox_inches='tight')

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
sns.boxplot(x='Promotion', y='SalesAmount', data=df, ax=axes[0])
axes[0].set_title('Влияние промо-акций на продажи')
sns.boxplot(x='HolidayMonth', y='SalesAmount', data=df, ax=axes[1])
axes[1].set_title('Влияние праздничных месяцев на продажи')
plt.savefig('plot-2.png', dpi=150, bbox_inches='tight')


# Классическая аддитивная декомпозиция (период = 12)
decomp_classic = seasonal_decompose(df['SalesAmount'], model='additive', period=12)
decomp_classic.plot()
plt.suptitle('Классическая аддитивная декомпозиция', y=1.02)
plt.savefig('plot-classic-add-decomposition.png', dpi=150, bbox_inches='tight')

# STL-декомпозиция (более устойчива к выбросам)
stl = STL(df['SalesAmount'], period=12, robust=True)
res = stl.fit()
res.plot()
plt.suptitle('STL-декомпозиция', y=1.02)
plt.savefig('plot-stl-composition.png', dpi=150, bbox_inches='tight')

# Быстрое преобразование Фурье
y = df['SalesAmount'].values
y = y - np.mean(y) # Убираем константу (тренд)

N = len(y)
yf = fft(y)
xf = fftfreq(N, 1) # Частота (1/месяц)

# Только положительные частоты
idx = np.where(xf > 0)
freqs = xf[idx]
power = np.abs(yf[idx])**2

# Перевод частоты в периоды (месяцы)
periods = 1 / freqs

plt.figure(figsize=(10, 5))
plt.stem(periods, power, basefmt=" ")
plt.xlabel('Период (месяцы)')
plt.ylabel('Мощность спектра')
plt.title('Спектральный анализ (FFT)')
plt.xlim(0, 24)
plt.grid(True)
plt.savefig('plot-furie.png', dpi=150, bbox_inches='tight')

# Непрерывное вейвлет-преобразование (CWT) с вейвлетом Морле
scales = np.arange(1, 32)
coefficients, frequencies = pywt.cwt(df['SalesAmount'], scales, 'morl')

plt.figure(figsize=(14, 6))
plt.imshow(np.abs(coefficients), extent=[0, len(df), 1, 32], cmap='jet', aspect='auto', vmax=abs(coefficients).max(), vmin=-abs(coefficients).max())
plt.colorbar(label='Амплитуда')
plt.title('Вейвлет-скалограмма (Вейвлет Морле)')
plt.xlabel('Время (месяцы)')
plt.ylabel('Масштаб (период)')
plt.savefig('plot-cwt.png', dpi=150, bbox_inches='tight')

