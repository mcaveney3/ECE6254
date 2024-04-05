
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
avg_vec = np.zeros((2216,1))
k=0
filtered_df = x_train[x_train['age_group'] == '18 - 34']
idx = filtered_df.index
for i in idx[0:2216]:
    #grab 18-34 y/o athlete and assign color for gender
    if (filtered_df['gender'][i]== 'M'):
        color = 'bo'
    else:
        color = 'ro'
    ath = filtered_df.iloc[i].athlete
    ath_data = filtered_df[(filtered_df['athlete']==ath) & (filtered_df['distance']!=0)]

    pace = ath_data['duration']/ath_data['distance']
    avg = np.average(pace)
    avg_vec[k] = avg; k+=1
    plt.plot(i,avg,color)

plt.ylim(0, 20)
print("mean min/km for age group: " + str(np.average(avg_vec)))