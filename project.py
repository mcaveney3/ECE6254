
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%% Read and separate data
d_data = pd.read_parquet('dataset/run_ww_2019_d.parquet', engine='pyarrow')
m_data = pd.read_parquet('dataset/run_ww_2019_m.parquet', engine='pyarrow')

# separate training data from test data
# 37598 athletes total, take first 14000- 7k for train, 7k for validation
train_idx = range(7000)
val_idx = range(7000,14000)

x_train = d_data[d_data['athlete'].isin(train_idx)]
x_val = d_data[d_data['athlete'].isin(val_idx)]
x_test = d_data[~d_data['athlete'].isin(val_idx) & ~d_data['athlete'].isin(train_idx)]


#%% Generate new features in training set from day data

#%% plotting fun!

n=1000
avg_data = np.zeros((n,1))
for i in range(n):
    #grab athlete and assign color for gender
    if (d_data['gender'][i]== 'M'):
        color = 'bo'
    else:
        color = 'ro'
    ath = x_train[(x_train['athlete']==i) & (x_train['duration']!=0)]
    if (ath['age_group'] !=	'18 - 34'):
        continue
    
    pace = ath['distance']/ath['duration']
    avg = np.average(pace)
    plt.plot(i,avg,color)