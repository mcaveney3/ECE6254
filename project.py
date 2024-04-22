
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from keras.regularizers import l2
from keras.layers import Dropout
from keras.optimizers import Adam
from imblearn.over_sampling import SMOTE
from matplotlib.lines import Line2D

#%% Read and separate data
d_data = pd.read_parquet('dataset/run_ww_2019_d.parquet')
m_data = pd.read_parquet('dataset/run_ww_2019_m.parquet')
q_data = pd.read_parquet('dataset/run_ww_2019_q.parquet')

# separate training data from test data
# 37598 athletes total, take first 14000- 7k for train, 7k for validation
train_idx = range(7000)
val_idx = range(7000,14000)

x_train = d_data[d_data['athlete'].isin(train_idx)]
x_val = d_data[d_data['athlete'].isin(val_idx)]
x_test = d_data[~d_data['athlete'].isin(val_idx) & ~d_data['athlete'].isin(train_idx)]


#%% Generate new features in training and validation set from day data for each athlete
marathons_2019 = ['BERLIN 2019','BOSTON 2019','LONDON 2019','CHICAGO 2019','TOKYO 2019','NEW YORK 2019']
race_days = [271,104,117,285,61,306]

for trainValn in range(2):
    x_set = x_train.copy() if trainValn == 1 else x_val.copy()
    
    enum_set = []
    enum_countries = list(set(x_set.country))
    #enum_major = list(set(x_set.major))
    enum_age = ['18 - 34', '35 - 54', '55 +']
    enum_gender = ['M', 'F']
    
    #calculate all values per athlete
    n = int(len(x_set)/365)
    idx = x_set.index
    
    for i in idx[0:n]:
        ath = x_set.iloc[i].athlete
        ath_data = x_set[(x_set['athlete']==ath)].copy().reset_index()
        
        #which 2019 marathon did they run (if any) ?
        marathon = ath_data.iloc[0].major
        day_of_race = next((race_days[i] for i in range(6) if marathon.find(marathons_2019[i]) != -1), -1)
        if day_of_race == -1:
            continue
        
        #athlete should have ran one of the 2019 marathons
        nonzero_idx4 = (ath_data['duration'] != 0) & (ath_data.index < day_of_race) & (ath_data.index >= day_of_race-28)
        nonzero_idx7 = nonzero_idx4 & (ath_data.index >= day_of_race-7)
        
        dist_last_7 = np.sum(ath_data['distance'][day_of_race-7:day_of_race])
        KPW_last_4 = np.sum(ath_data['distance'][day_of_race-28:day_of_race])/4
        dur_last_7 = np.sum(ath_data['duration'][day_of_race-7:day_of_race])
        HPW_last_4 = np.sum(ath_data['duration'][day_of_race-28:day_of_race])/4
        pace_var_last_7 = np.nan_to_num(np.var(ath_data.loc[nonzero_idx7].duration/ath_data.loc[nonzero_idx7].distance))*100
        pace_var_last_4 = np.nan_to_num(np.var(ath_data.loc[nonzero_idx4].duration/ath_data.loc[nonzero_idx4].distance))*100
        LR_last_7 = np.sum((ath_data['distance'][day_of_race-7:day_of_race] > 13.1) == True)
        LR_last_4 = np.sum((ath_data['distance'][day_of_race-28:day_of_race] > 13.1) == True)
        Longest_run_7 = max(ath_data['distance'][day_of_race-7:day_of_race])
        Longest_run_4 = max(ath_data['distance'][day_of_race-28:day_of_race])

        
        #only take non-zero runs for rest of calculations
        ath_data = x_set[(x_set['athlete']==ath) & (x_set['distance']!=0)]
        dist = ath_data['distance']
        dur = ath_data['duration']
        pace = dur/dist
        
        new_entry = {
            'athlete' : ath,
            'distance_avg' : np.average(dist), #in full year
            'distance_total' : np.sum(dist), #in full year
            'duration_avg' : np.average(dur), #in full year
            'duration_total' : np.sum(dur), #in full year
            'gender' : enum_gender.index(ath_data.iloc[0].gender),
            'age_group' : enum_age.index(ath_data.iloc[0].age_group),
            #'country' : enum_countries.index(ath_data.iloc[0].country),
            #'major' : enum_major.index(ath_data.iloc[0].major),
            #'pace_avg' : np.average(pace),
            'runs_per_week' : len(ath_data)/52,
            #'distance_variance' : np.var(dist),
            #'duration_variance' : np.var(dur),
            
            #all data leading up to athlete's marathon
            'dist_last_7_days' : dist_last_7,
            'KPW_last_4_weeks' : KPW_last_4,
            'duration_last_7_days' : dur_last_7,
            'HPW_last_4_weeks' : HPW_last_4,
            'pace_var_last_7_days' : pace_var_last_7,
            'pace_var_last_74_weeks' : pace_var_last_4,
            'long_runs_last_7_days' : LR_last_7,
            'long_runs_last_4_weeks' : LR_last_4,
            'longest_run_last_7_days' : Longest_run_7,
            'longest_run_last_4_weeks': Longest_run_4
            #'pace_variance' : np.var(pace)
            }
        
        enum_set.append(new_entry)
        
    if trainValn == 1:
        enum_train_df = pd.DataFrame(enum_set)
        
        #normalize all columns except gender and athlete ID
        #scaler = StandardScaler()
        #enum_train_df_norm = enum_train_df.copy()
        #enum_df_num = enum_train_df_norm.drop(['athlete', 'gender'], axis=1)
        #enum_train_df_norm[enum_df_num.columns] = scaler.fit_transform(enum_df_num)
    else:
        enum_val_df = pd.DataFrame(enum_set)
        
        #normalize all columns except gender and athlete ID
        scaler = StandardScaler()
        enum_val_df_norm = enum_val_df.copy()
        enum_df_num = enum_val_df_norm.drop(['athlete', 'gender'], axis=1)
        enum_val_df_norm[enum_df_num.columns] = scaler.fit_transform(enum_df_num)
        

#%% resample and normalize
X_train_orig = enum_train_df.drop(['athlete', 'gender'], axis=1).copy()
y_train_orig = enum_train_df[['gender']]
y_val = enum_val_df[['gender']]

#resample
sm = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = sm.fit_resample(X_train_orig, y_train_orig)

#normalize all columns except gender and athlete ID for train data
scaler = StandardScaler()
X_train = X_train_resampled.copy()
X_train[X_train.columns] = scaler.fit_transform(X_train)
y_train = y_train_resampled

#%% logistic regression

# Fit the model with the training data
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train.gender)

# Predicting the Test set results
y_pred = logistic_model.predict(enum_val_df_norm.drop(['athlete', 'gender'], axis=1))
LR_accuracy = accuracy_score(y_val, y_pred)
LR_report = classification_report(y_val, y_pred)


#%% LDA
lda = LDA(n_components=1)
lda.fit(X_train, y_train.gender)
y_pred = lda.predict(enum_val_df_norm.drop(['athlete', 'gender'], axis=1))
LDA_accuracy = accuracy_score(y_val, y_pred)
LDA_report = classification_report(y_val, y_pred)

#%% neural net
learning_rate = 0.001
optimizer = Adam(learning_rate=learning_rate)

nn_model = Sequential([
    Dense(15, input_shape=(X_train.shape[1],), activation='relu', kernel_regularizer=l2(0.01)),
    #Dropout(0.5),
    Dense(10, activation='sigmoid', kernel_regularizer=l2(0.01)),  
    #Dropout(0.5),
    Dense(1, activation='sigmoid')
])

nn_model.compile(optimizer=optimizer,
              loss='binary_crossentropy',
              metrics=['accuracy', 'Recall','Precision'])
history = nn_model.fit(X_train, y_train.gender, epochs=30, validation_data=(enum_val_df_norm.drop(['athlete', 'gender'], axis=1), y_val.gender))
nn_loss, nn_accuracy, nn_recall, nn_precision = nn_model.evaluate(enum_val_df_norm.drop(['athlete', 'gender'], axis=1), y_val.gender)

#%% plotting separated by age for M/F
k=0
age_group= '18 - 34'
filtered_df = x_train[(x_train['age_group'] == age_group) & (x_train['major'].str.contains('CHICAGO 2019'))]
#country = 'United Kingdom'
#filtered_df = x_train[(x_train['country'] == country)]

'''
idx = filtered_df.index
n = int(len(filtered_df)/365)
m_avg_vec = np.zeros((n,1))
f_avg_vec = np.zeros((n,1))


for i in idx[0:n]:
    #grab 18-34 y/o athlete and assign color for gender
    ath = filtered_df.iloc[i].athlete
    ath_data = filtered_df[(filtered_df['athlete']==ath) & (filtered_df['distance']!=0)]

    pace = ath_data['duration']/ath_data['distance']
    avg = np.average(pace)
    #dist = ath_data['distance']
    #avg = np.average(dist)
    
    if (filtered_df['gender'][i]== 'M'):
        color = 'b.'
        m_avg_vec[k] = avg; k+=1
    else:
        color = 'r.'
        f_avg_vec[k] = avg; k+=1
    
    plt.plot(ath_data.index,pace,color)
'''

mask = (filtered_df['distance'] != 0) & (filtered_df['gender'] == 'M')
subset = filtered_df[mask]
color_metric = subset['duration'] / subset['distance']

fig, ax = plt.subplots()
plt.axvline(pd.to_datetime('2019-10-13 00:00:00'), color='r')
scatter = ax.scatter(subset['datetime'], subset['distance'], c=color_metric, s=1,vmin=5,vmax=8, cmap='inferno')
plt.colorbar(scatter, ax=ax, label='Pace (min/km)')
ax.set_xlabel('Date')
ax.set_ylabel('Distance (km)')
ax.set_title('2019 Chicago Marathon - 18-34 Year Old Female Training Data')

plt.show()

#%% plotting pace by age group
fig, ax = plt.subplots()
k=0
#filtered_df = x_train[x_train.gender == 'M']
filtered_df = x_train

idx = filtered_df.index
n = int(len(filtered_df)/365)
avg_vec_1 = np.zeros((n,1))
avg_vec_2 = np.zeros((n,1))
avg_vec_3 = np.zeros((n,1))

for i in idx[0:n]:
    #grab athlete and assign color for age
    ath = filtered_df.iloc[i].athlete
    ath_data = filtered_df[(filtered_df['athlete']==ath) & (filtered_df['distance']!=0)]

    pace = np.sum(ath_data['duration'])/np.sum(ath_data['distance'])
    if (pace > 1000):
        print(str(i))
    #avg = np.average(pace)
    #dist = ath_data['distance']
    #avg = np.average(dist)
    
    if (filtered_df['age_group'][i] == '18 - 34'):
        color = 'blue'
        avg_vec_1[k] = pace; k+=1
    elif (filtered_df['age_group'][i] == '35 - 54'):
        color = 'red'
        avg_vec_2[k] = pace; k+=1
    else:
        avg_vec_3[k] = pace; k+=1
        color = 'green'
    
    ax.scatter(i,pace,color = color,s=1)
    
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='18 - 34', markerfacecolor='blue', markersize=5),
    Line2D([0], [0], marker='o', color='w', label='35 - 54', markerfacecolor='red', markersize=5),
    Line2D([0], [0], marker='o', color='w', label='55 +', markerfacecolor='green', markersize=5)
]
ax.legend(handles=legend_elements)
plt.ylim(0,15)
ax.set_xlabel('Athlete #')
ax.set_ylabel('Average Pace (min/km)')
ax.set_title('Average Pace by Age')





