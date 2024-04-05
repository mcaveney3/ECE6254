
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

filtered_df = x_train[x_train['age_group'] == '18 - 34']
idx = filtered_df.index
for i in idx:
    #grab athlete and assign color for gender
    if (filtered_df['gender'][i]== 'M'):
        color = 'bo'
    else:
        color = 'ro'
    ath = filtered_df[(filtered_df['athlete']==i) & (filtered_df['duration']!=0)]
    
    pace = ath['distance']/ath['duration']
    avg = np.average(pace)
    plt.plot(i,avg,color)