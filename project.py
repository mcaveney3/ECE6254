
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

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
enum_set = []
enum_countries = list(set(x_train.country))
#enum_major = list(set(x_train.major))
enum_age = list(set(x_train.age_group))
enum_gender = list(set(x_train.gender))

#calculate all values per athlete
n = int(len(x_train)/365)
idx = x_train.index
for i in idx[0:n]:
    ath = x_train.iloc[i].athlete
    ath_data = x_train[(x_train['athlete']==ath) & (x_train['distance']!=0)]

    dist = ath_data['distance']
    dur = ath_data['duration']
    pace = dur/dist
    
    new_entry = {
        'athlete' : ath,
        'distance_avg' : np.average(dist),
        'duration_avg' : np.average(dur),
        'gender' : enum_gender.index(ath_data.iloc[0].gender),
        'age_group' : enum_age.index(ath_data.iloc[0].age_group),
        'country' : enum_countries.index(ath_data.iloc[0].country),
        #'major' : enum_major.index(ath_data.iloc[0].major),
        'pace_avg' : np.average(pace),
        'runs_per_week' : len(ath_data)/52,
        'distance_variance' : np.var(dist),
        'duration_variance' : np.var(dur),
        'pace_variance' : np.var(pace)
        }
    
    enum_set.append(new_entry)

enum_df = pd.DataFrame(enum_set)

#normalize all columns except gender and athlete ID
scaler = StandardScaler()
enum_df_norm = enum_df
enum_df_num = enum_df_norm.drop(['athlete', 'gender'], axis=1)
enum_df_norm[enum_df_num.columns] = scaler.fit_transform(enum_df_num)

#%% plotting separated by age for M/F

k=0
#age_group= '55 +'
#filtered_df = x_train[x_train['age_group'] == age_group]
country = 'United Kingdom'
filtered_df = x_train[(x_train['country'] == country)]


idx = filtered_df.index
n = int(len(filtered_df)/365)
m_avg_vec = np.zeros((n,1))
f_avg_vec = np.zeros((n,1))

for i in idx[0:n]:
    #grab 18-34 y/o athlete and assign color for gender
    ath = filtered_df.iloc[i].athlete
    ath_data = filtered_df[(filtered_df['athlete']==ath) & (filtered_df['distance']!=0)]

    #pace = ath_data['distance']/ath_data['duration']
    #avg = np.var(pace)
    dist = ath_data['distance']
    avg = np.average(dist)
    
    if (filtered_df['gender'][i]== 'M'):
        color = 'bo'
        m_avg_vec[k] = avg; k+=1
    else:
        color = 'ro'
        f_avg_vec[k] = avg; k+=1
    
    plt.plot(i,avg,color)

#plt.ylim(0, 100)
print("country: " + country)
print("avg dist overall: " + str(np.average(m_avg_vec + f_avg_vec)))
print("avg dist male: " + str(np.average(m_avg_vec[m_avg_vec!=0])))
print("avg dist female: " + str(np.average(f_avg_vec[f_avg_vec!=0])))







