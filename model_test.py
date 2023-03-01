import pandas as pd
import pickle


test = pd.read_csv("C:\\Users\\GauravUgale\\Desktop\\test.csv")

x_columns = ['mean', 'std_dev', 'kurtosis', 'median', 'zero_crossings', 'slope_changes',
'max_freq_amp', 'peak_freq', 'spectral_skewness', 'spectral_kurtosis', 'std', 
'correlate', 'mean_absolute_percentage_error']

x = test[x_columns].values

model = pickle.load(open('stc_new_1.pickle', 'rb'))

test['Label'] = model.predict(x)

test.to_csv("C:\\Users\\GauravUgale\\Desktop\\test_label2.csv", index=False)