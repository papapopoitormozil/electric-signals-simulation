import numpy as np
import matplotlib.pyplot as plt

fs = 1000
duration = 2.0
t = np.arange(0, duration, 1/fs)
f_signal = 10
f_interference = 50
noise_std = 0.20
window = 20

signal = np.sin(2*np.pi*f_signal*t)
rng = np.random.default_rng(2026)
white_noise = rng.normal(0, noise_std, len(t))

def moving_average(x, n):
    return np.convolve(x, np.ones(n)/n, mode="same")

def snr_db(reference, noise):
    return 10*np.log10(np.mean(reference**2)/np.mean(noise**2))

clean_filtered = moving_average(signal, window)

for amplitude in [0.1, 0.25, 0.5, 0.75, 1.0]:
    interference = amplitude*np.sin(2*np.pi*f_interference*t)
    noisy = signal + interference + white_noise
    filtered = moving_average(noisy, window)

    snr_before = snr_db(signal, noisy-signal)
    residual_noise = filtered-clean_filtered
    snr_after = snr_db(clean_filtered, residual_noise)

    rmse_before = np.sqrt(np.mean((noisy-signal)**2))
    rmse_after = np.sqrt(np.mean((filtered-signal)**2))
    print(amplitude, snr_before, snr_after, rmse_before, rmse_after)

# amplitude 0.5
interference = 0.5*np.sin(2*np.pi*f_interference*t)
noisy = signal + interference + white_noise
filtered = moving_average(noisy, window)
plt.plot(t[:250], signal[:250], label="Полезный сигнал")
plt.plot(t[:250], noisy[:250], label="С помехой")
plt.plot(t[:250], filtered[:250], label="После фильтрации")
plt.xlabel("Время, с"); plt.ylabel("Амплитуда, усл. ед.")
plt.grid(); plt.legend(); plt.show()
