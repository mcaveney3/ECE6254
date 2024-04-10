
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

#%% Read and separate data
#d_data = pd.read_parquet('dataset/run_ww_2019_d.parquet', engine='pyarrow')
w_data = pd.read_parquet('dataset/run_ww_2019_w.parquet', engine='pyarrow')
#m_data = pd.read_parquet('dataset/run_ww_2019_m.parquet', engine='pyarrow')

# separate training data from test data
# 5537 monthly runners, take 1250 for training, 1250 for validation and rest for test
# pruning all datasets to weekly runners
iw = w_data[w_data.distance>0].athlete.value_counts()
weekly_runners = iw.index[iw.gt(51)].tolist()

train_idx, test_idx = train_test_split(weekly_runners, train_size=2500)
train_idx, val_idx = train_test_split(train_idx, train_size=1250)
        
x_train = w_data[w_data['athlete'].isin(train_idx)]
x_val = w_data[w_data['athlete'].isin(val_idx)]
x_test = w_data[~w_data['athlete'].isin(val_idx) & ~w_data['athlete'].isin(train_idx)]

#%% Generate new features in training and validation set from day data
for trainValn in range(2):
    x_set = x_train.copy() if trainValn == 1 else x_val.copy()
    
    enum_set = []
    enum_countries = list(set(x_set.country))
    enum_age = list(set(x_set.age_group))
    enum_gender = list(set(x_set.gender))
    
    #calculate all values per athlete per week
    n = int(len(x_set)/52)
    idx = x_set.index
    for i in idx[0:n]:
        ath = x_set.iloc[i].athlete
        ath_data = x_set[(x_set['athlete']==ath) & (x_set['distance']!=0)]
        
        dist = ath_data['distance']
        dur = ath_data['duration']
        pace = dur/dist
        
        new_entry = {
            'athlete' : ath,
            'distance_avg' : np.average(dist),
            'duration_avg' : np.average(dur),
            'gender' : enum_gender.index(ath_data.iloc[0].gender),
            'age_group' : enum_age.index(ath_data.iloc[0].age_group),
            'pace_avg' : np.average(pace),
            'distance_variance' : np.var(dist),
            'duration_variance' : np.var(dur),
            'pace_variance' : np.var(pace),
            }
        
        for j in range(0,4):
            week_idx = ath_data.index[pd.to_datetime(ath_data.datetime).dt.isocalendar().week % 4 == j]
            week_ath_data = ath_data[ath_data.index.isin(week_idx)]

            dist_diff = abs(ath_data.distance.diff()).tolist()
            dist_diff[0] = 0
            dur_diff = abs(ath_data.duration.diff()).tolist()
            dur_diff[0] = 0
            pace_diff = abs(pace.diff()).tolist()
            pace_diff[0] = 0
            
            s = 'week_' + str(j) + '_'
            
            new_entry[s + 'distance_diff_avg'] = np.average(dist_diff)
            new_entry[s + 'distance_diff_variance'] = np.var(dist_diff)
            new_entry[s + 'duration_diff_avg'] = np.average(dur_diff)
            new_entry[s + 'duration_diff_variance'] = np.var(dur_diff)
            new_entry[s + 'pace_diff_avg'] = np.average(pace_diff)
            new_entry[s + 'pace_diff_variance'] = np.var(pace_diff)
            
        enum_set.append(new_entry)
        
    if trainValn == 1:
        enum_train_df = pd.DataFrame(enum_set)
        
        #normalize all columns except gender and athlete ID
        scaler = StandardScaler()
        enum_train_df_norm = enum_train_df.copy()
        enum_df_num = enum_train_df_norm.drop(['athlete', 'gender'], axis=1)
        enum_train_df_norm[enum_df_num.columns] = scaler.fit_transform(enum_df_num)
    else:
        enum_val_df = pd.DataFrame(enum_set)
        
        #normalize all columns except gender and athlete ID
        scaler = StandardScaler()
        enum_val_df_norm = enum_val_df.copy()
        enum_df_num = enum_val_df_norm.drop(['athlete', 'gender'], axis=1)
        enum_val_df_norm[enum_df_num.columns] = scaler.fit_transform(enum_df_num)


#%% logistic regression
y_train = enum_train_df[['gender']]
y_val = enum_val_df[['gender']]

# Fit the model with the training data
logistic_model = LogisticRegression(class_weight={0:0.7, 1:0.3})#, solver="liblinear")
logistic_model.fit(enum_train_df_norm.drop(['athlete', 'gender'], axis=1), y_train.gender)

# Predicting the Test set results
y_pred = logistic_model.predict(enum_val_df_norm.drop(['athlete', 'gender'], axis=1))
accuracy = accuracy_score(y_val, y_pred)
report = classification_report(y_val, y_pred)

print('Logistic Regression:')
print( 'Accuracy: ' + str( accuracy ))
print( report )


#LDA
lda = LDA(n_components=1)
lda.fit(enum_train_df_norm.drop(['athlete', 'gender'], axis=1), y_train.gender)
y_pred = lda.predict(enum_val_df_norm.drop(['athlete', 'gender'], axis=1))
LDA_accuracy = accuracy_score(y_val, y_pred)
LDA_report = classification_report(y_val, y_pred)
print('LDA:')
print( 'Accuracy: ' + str( LDA_accuracy ))
print( LDA_report )


