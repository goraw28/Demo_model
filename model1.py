# import numpy as np
# import scipy.stats as stats
# from scipy.signal import welch, detrend
# import pandas as pd
# from time import time
# import glob

# def signal_properties(signal, fs=250, nperseg=1024):
#     # Calculate basic properties
#     mean = np.mean(signal)
#     std_dev = np.std(signal)
#     min_val = np.min(signal)
#     max_val = np.max(signal)
#     range_val = max_val - min_val
    
#     # Calculate statistical properties
#     skewness = stats.skew(signal)
#     kurtosis = stats.kurtosis(signal)
#     median = np.median(signal)
#     mode = stats.mode(signal, keepdims=True)[0][0]
    
#     # Calculate time-domain properties
#     signal_diff = np.diff(signal)
#     zero_crossings = len(np.where(np.diff(np.sign(signal)))[0])
#     slope_changes = len(np.where(np.diff(np.sign(signal_diff)))[0])
    
#     # Calculate frequency-domain properties using FFT
#     fft_signal = np.fft.fft(signal)
#     power_spectrum = np.abs(fft_signal) ** 2
#     freqs = np.fft.fftfreq(len(signal))
#     max_freq_amp = np.max(power_spectrum)
#     peak_freq = freqs[np.argmax(power_spectrum)]

#     # Calculate the spectral centroid, skewness, bandwidth, kurtosis
#     f, psd = welch(signal, fs=fs, nperseg=nperseg)
#     spectral_centroid = np.sum(f * psd) / np.sum(psd)
#     spectral_bandwidth = np.sqrt(np.sum(psd * (f - spectral_centroid) ** 2) / np.sum(psd))
#     spectral_skewness = np.sum(psd * (f - spectral_centroid) ** 3) / np.sum(psd) / spectral_bandwidth ** 3
#     spectral_kurtosis = (np.sum(psd**4) / len(psd)) / (psd.var()**2) - 2*(stats.kurtosis(psd) - 3)
    
    
#     # Create a dictionary to store the properties
#     properties = {
#         'mean': mean,
#         'std_dev': std_dev,
#         'min_val': min_val,
#         'max_val': max_val,
#         'range_val': range_val,
#         'skewness': skewness,
#         'kurtosis': kurtosis,
#         'median': median,
#         'mode': mode,
#         'zero_crossings': zero_crossings,
#         'slope_changes': slope_changes,
#         'max_freq_amp': max_freq_amp,
#         'peak_freq': peak_freq,
#         'spectral_centroid': spectral_centroid, 
#         'spectral_bandwidth': spectral_bandwidth, 
#         'spectral_skewness': spectral_skewness,
#         'spectral_kurtosis': spectral_kurtosis
#     }
    
#     values = [mean, std_dev, min_val, max_val, range_val, skewness, 
#               kurtosis, median, mode, zero_crossings, slope_changes, 
#               max_freq_amp, peak_freq, spectral_centroid,
#               spectral_bandwidth, spectral_skewness, spectral_kurtosis]
    
#     return values

# if __name__=="__main__":
    
#     import os
#     from tqdm import tqdm
    
#     path = "C:\\Users\\x\\Desktop\\x\\x\\"
    
#     files = [f for f in os.listdir(path) if f.endswith('.csv')]
    
#     # df = pd.read_csv('C:\\Users\\x\\Desktop\\x\\x.csv')
    
#     columns = ['FileName', 'mean', 'std_dev', 'min_val', 'max_val', 'range_val', 'skewness', 
#                'kurtosis', 'median', 'mode', 'zero_crossings', 'slope_changes', 
#                'max_freq_amp', 'peak_freq', 'spectral_centroid',  
#                'spectral_bandwidth',  'spectral_skewness', 'spectral_kurtosis']  
    
#     all_vals = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]).reshape(1,-1)
    
#     # for filename in glob.glob("C:\\Users\\x\\Desktop\\x\\x\\*"):
#     #     name = filename.split("\\")[-1].split(".")[0]
#     #     y = pd.read_csv(name).iloc[:,-1].values

#     for file in tqdm(files):
#         try:
#             s = pd.read_csv(f'{path}{file}').iloc[:,-1].values
#             s_shift = detrend(s, type='linear')
#             val = signal_properties(s_shift)
#             val = [file] + val
#             all_vals = np.append(all_vals, [val], axis=0)
#         except:
#             pass
        
#     pd.DataFrame(all_vals[1:], columns=columns).to_csv('signal_properties1.csv', index=False)



import numpy as np
import scipy.stats as stats
from scipy.signal import welch, detrend
import pandas as pd
from statistics import pvariance
from sklearn.preprocessing import MinMaxScaler
from time import time
import os
from tqdm import tqdm
from sklearn.metrics import mean_absolute_percentage_error, d2_tweedie_score, d2_pinball_score, max_error


def normalize(signal, feature_range=(0,1)):
    scaler = MinMaxScaler(feature_range=feature_range)
    if signal.ndim==2: 
        a = scaler.fit_transform(signal)
        return a.reshape(len(a),)
    a = scaler.fit_transform(signal.reshape(-1,1))
    return a.reshape(len(a),)

def fft_clear_signal(signal, dt=1, low_thr=0., high_thr=0.5, power=True):
    n = len(signal)
    fhat = np.fft.fft(signal, n)
    PSD = fhat * np.conj(fhat)/n
    freq = (1/(dt*n))*np.arange(n)
    L = np.arange(1, np.floor(n/2), dtype='int')
    if power:
        indices = PSD > high_thr
    else:
        indices = np.where(((low_thr <= freq) & (freq <= high_thr)), True, False)
    PSDclean = PSD * indices
    fhat = indices * fhat
    ffilt = np.fft.ifft(fhat)
    return ffilt.real

def signal_matrix(signal):
    s_norm = normalize(signal)
    # gradient
    gradient = np.gradient(s_norm)
    
    # standard deviation of gradient
    gradient_std = np.std(gradient)
    
    # differentiate of gradient and original signal
    gradient_pVar = (pvariance(gradient) / pvariance(s_norm))
    
    std = np.std(s_norm)
            
    clean_signal = fft_clear_signal(s_norm, low_thr=0, high_thr=0.15, power=False)
    
    cl_norm = normalize(clean_signal)
    
    ## correlation coefficients
    # coff = np.corrcoef(cl_norm, s_norm)[0,1]
    
    ## population variance
    pVar = (pvariance(cl_norm) / pvariance(s_norm))
    
    ## Cross-correlation
    correlate = np.correlate(cl_norm, s_norm, mode='valid')[0]
    
    ## kurtosis
    kur = stats.kurtosis(s_norm, fisher=False)
    
    
    metrics = [mean_absolute_percentage_error, max_error, d2_tweedie_score, d2_pinball_score]
    val_list = []
    
    for m in metrics:
        val_list.append(m(s_norm, gradient))
    
    return [gradient_std, gradient_pVar, std, pVar, correlate] + val_list + [kur]
'''
'gradient_std':[], 'gradient_pVar':[], 'std':[], 'pVar':[], 'correlate':[], 
            'mean_absolute_percentage_error':[], 'max_error':[], 'd2_tweedie_score':[], 'd2_pinball_score':[], 
            'kurtosis':[],
'''
def signal_properties(signal, fs=250, nperseg=1024):
    # Calculate basic properties
    mean = np.mean(signal)
    std_dev = np.std(signal)
    min_val = np.min(signal)
    max_val = np.max(signal)
    range_val = max_val - min_val
    
    # Calculate statistical properties
    skewness = stats.skew(signal)
    kurtosis = stats.kurtosis(signal)
    median = np.median(signal)
    mode = stats.mode(signal, keepdims=True)[0][0]
    
    # Calculate time-domain properties
    signal_diff = np.diff(signal)
    zero_crossings = len(np.where(np.diff(np.sign(signal)))[0])
    slope_changes = len(np.where(np.diff(np.sign(signal_diff)))[0])
    
    # Calculate frequency-domain properties using FFT
    fft_signal = np.fft.fft(signal)
    power_spectrum = np.abs(fft_signal) ** 2
    freqs = np.fft.fftfreq(len(signal))
    max_freq_amp = np.max(power_spectrum)
    peak_freq = freqs[np.argmax(power_spectrum)]

    # Calculate the spectral centroid, skewness, bandwidth, kurtosis
    f, psd = welch(signal, fs=fs, nperseg=nperseg)
    spectral_centroid = np.sum(f * psd) / np.sum(psd)
    spectral_bandwidth = np.sqrt(np.sum(psd * (f - spectral_centroid) ** 2) / np.sum(psd))
    spectral_skewness = np.sum(psd * (f - spectral_centroid) ** 3) / np.sum(psd) / spectral_bandwidth ** 3
    spectral_kurtosis = (np.sum(psd**4) / len(psd)) / (psd.var()**2) - 2*(stats.kurtosis(psd) - 3)   
    
    # Create a dictionary to store the properties
    properties = {
        'mean': mean,
        'std_dev': std_dev,
        'min_val': min_val,
        'max_val': max_val,
        'range_val': range_val,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'median': median,
        'mode': mode,
        'zero_crossings': zero_crossings,
        'slope_changes': slope_changes,
        'max_freq_amp': max_freq_amp,
        'peak_freq': peak_freq,
        'spectral_centroid': spectral_centroid, 
        'spectral_bandwidth': spectral_bandwidth, 
        'spectral_skewness': spectral_skewness,
        'spectral_kurtosis': spectral_kurtosis
    }
    
    values = [mean, std_dev, min_val, max_val, range_val, skewness, 
              kurtosis, median, mode, zero_crossings, slope_changes, 
              max_freq_amp, peak_freq, spectral_centroid,
              spectral_bandwidth, spectral_skewness, spectral_kurtosis]
    
    return values

if __name__=="__main__":
    

    
    path = "C:\\Users\\x\\Downloads\\x\\x\\"
    
    files = [f for f in os.listdir(path) if f.endswith('.csv')]
    
    # df = pd.read_csv('label.csv')
    
    columns = ['FileName', 'mean', 'std_dev', 'min_val', 'max_val', 'range_val', 'skewness', 
               'kurtosis', 'median', 'mode', 'zero_crossings', 'slope_changes', 
               'max_freq_amp', 'peak_freq', 'spectral_centroid', 'spectral_bandwidth',  
               'spectral_skewness', 'spectral_kurtosis', 'gradient_std', 'gradient_pVar', 
               'std', 'pVar', 'correlate', 'mean_absolute_percentage_error', 'max_error', 
               'd2_tweedie_score', 'd2_pinball_score', 'kurtosis']  
    
    all_vals = np.array(range(len(columns))).reshape(1,-1)
    
    
    for file in tqdm(files):
        s = pd.read_csv(f'{path}{file}').iloc[:,-1].values
        s_shift = detrend(s, type='linear')
        val = [file] + signal_properties(s_shift) + signal_matrix(s)
        all_vals = np.append(all_vals, [val], axis=0)
        
    pd.DataFrame(all_vals[1:], columns=columns).to_csv('xxx.csv', index=False)