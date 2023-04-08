#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Generic Imports
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

plt.style.use('ggplot')
plt.rcParams["figure.figsize"] = (15,20)
#pd.set_option('precision', 3)
pd.set_option('display.max_columns',50)
np.set_printoptions(precision=3)

# Project Specific Imports
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split,GridSearchCV


# ## Import Data

# In[ ]:


f_name = 'ObesityDataSet_raw_and_data_sinthetic.csv'
df = pd.read_csv(f_name, header=0)


# In[ ]:


df.head()


# ## Data Prep
# 
# The data set attributes are as follows from the paper [found here](https://www.sciencedirect.com/science/article/pii/S2352340919306985?via%3Dihub)
# <br>
# <br>
# Frequent consumption of high caloric food (FAVC), Frequency of consumption of vegetables (FCVC), Number of main meals (NCP), Consumption of food between meals (CAEC), Consumption of water daily (CH20), and Consumption of alcohol (CALC). The attributes related with the physical condition are: Calories consumption monitoring (SCC), Physical activity frequency (FAF), Time using technology devices (TUE), Transportation used (MTRANS), other variables obtained were: Gender, Age, Height and Weight. Finally, all data was labeled and the class variable NObesity was created with the values of: Insufficient Weight, Normal Weight, Overweight Level I, Overweight Level II, Obesity Type I, Obesity Type II and Obesity Type III

# In[ ]:


# check for missing values
df.isna().sum()


# In[ ]:


df.info()


# In[ ]:


# Check cardinality of categorical features
num_features = tuple(df.select_dtypes(include=['float64']).columns)
cat_features = tuple(df.select_dtypes(include=['object']).columns)
for col in cat_features:
    print(f'{col} value counts')
    print(df[col].value_counts())
    print()


# ### Convert Binary Vars to 0/1

# In[ ]:


# Gender
df['Gender'] = (df.Gender == 'Male').astype(int)

# family_history_with_overweight
df.family_history_with_overweight = (df.family_history_with_overweight == 'yes').astype(int)

# FAVC
df.FAVC = (df.FAVC == 'yes').astype(int)

# SMOKE 
df.SMOKE = (df.SMOKE == 'yes').astype(int)

# SCC
df.SCC = (df.SCC == 'yes').astype(int)


# ### Encode Ordinal Vars

# In[ ]:


ord_encoders = {}
ord_vars = ('CAEC','CALC','NObeyesdad')
ord_vals = [(('no','Sometimes','Frequently','Always'),),
           (('no','Sometimes','Frequently','Always'),),
            (('Insufficient_Weight','Normal_Weight','Overweight_Level_I',
             'Overweight_Level_II','Obesity_Type_I','Obesity_Type_II','Obesity_Type_III'),)]
for i,key in enumerate(ord_vars):
    print(key,ord_vals[i])
    ord_encoders[key] = OrdinalEncoder(categories=ord_vals[i])
    ord_encoders[key].fit(np.asarray(df[key]).reshape(-1,1))
    col = 'ord_'+key
    df[col] = ord_encoders[key].transform(np.asarray(df[key]).reshape(-1,1))


# ### One-Hot Encode MTRANS

# In[ ]:


df = pd.concat([df,pd.get_dummies(df.MTRANS).add_prefix('MTRANS_')],axis=1)
df.head()


# ### Collect Final X vars

# In[ ]:


df.columns


# In[ ]:


x_cols = ['Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight',
          'FAVC', 'FCVC', 'NCP','SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE', 'ord_CAEC',
          'ord_CALC','MTRANS_Automobile', 'MTRANS_Bike', 'MTRANS_Motorbike',
          'MTRANS_Public_Transportation', 'MTRANS_Walking']
df[['ord_NObeyesdad'] + x_cols]


# ### Min Max Scale X Vars

# In[ ]:


scaler = MinMaxScaler()
df[x_cols] = scaler.fit_transform(df[x_cols])
df[x_cols].head()


# ## Test Train Split

# In[ ]:


X = df[x_cols]
y = df['ord_NObeyesdad']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f'X Train {X_train.shape}')
print(f'X Test {X_test.shape}')
print(f'y Train {y_train.shape}')
print(f'y test {y_test.shape}')


# ## EDA

# In[ ]:


pd.concat([y_train,X_train],axis=1).groupby('ord_NObeyesdad').mean()


# Looks like there is some differences in mean between several of the variables by qualitative observation. 

# In[ ]:


_=sns.heatmap(X_train.corr(), annot = True,linewidths=.5)


# Some of the transportation one-hot variables are highly correlated but that is to be expected. Generally, the variables looks decoupled.

# ## Model Selection

# In[ ]:


from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, precision_score,recall_score,f1_score,confusion_matrix, plot_confusion_matrix


mdls = {'lr': LogisticRegression(multi_class='ovr'),
        'svc': SVC(decision_function_shape='ovr'),
        'rf': RandomForestClassifier(n_jobs=-1),
        'xgb': XGBClassifier(n_jobs=-1),
        }

prms = {'lr': {'C': np.logspace(-4,4,9),
              'penalty': ['l1','l2']},
        'svc': {'C': np.logspace(-4,4,9)},
        'rf': {'n_estimators': [2**i for i in range(3,8)],
               'max_depth':  [8,16,32,64,None]},
        'xgb': {'min_child_weight': [4,5],
                'gamma': [i/10.0 for i in range(3,6)],
                'subsample': [i/10.0 for i in range(6,11)],
                'colsample_bytree': [i/10.0 for i in range(6,11)],
                'max_depth': [2,3,4]}
        }

# Commented out to aviod retraining
best_estimators = {}
for key in mdls.keys():
    print('Training model: {}'.format(key))
    gs_cv = GridSearchCV(mdls[key],prms[key], cv = 5,
                         scoring='f1_macro', n_jobs = -1, verbose=1)
    best_est = gs_cv.fit(X_train, y_train)
    print('Best Estimator: {}'.format(best_est.best_params_))
    print('Best Estimator f1-score: {}'.format(best_est.best_score_))
    best_estimators[key] = best_est.best_estimator_
# #### Results
# Training model: lr<br>
# Fitting 5 folds for each of 18 candidates, totalling 90 fits<br>
# Best Estimator: {'C': 10000.0, 'penalty': 'l2'}<br>
# Best Estimator f1-score: 0.7735656783439809<br>
# Training model: svc<br>
# Fitting 5 folds for each of 9 candidates, totalling 45 fits<br>
# Best Estimator: {'C': 1000.0}<br>
# Best Estimator f1-score: 0.91378212026485<br>
# Training model: rf<br>
# Fitting 5 folds for each of 25 candidates, totalling 125 fits<br>
# Best Estimator: {'max_depth': 64, 'n_estimators': 128}<br>
# Best Estimator f1-score: 0.951265750830309<br>
# Training model: xgb<br>
# Fitting 5 folds for each of 450 candidates, totalling 2250 fits<br>
# [22:51:46] WARNING: C:/Users/Administrator/workspace/xgboost-win64_release_1.4.0/src/learner.cc:1095: <br>Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'multi:softprob' <br>was changed from 'merror' to 'mlogloss'. Explicitly set eval_metric if you'd like to restore the <br>old behavior.<br>
# Best Estimator: {'colsample_bytree': 0.9, 'gamma': 0.4, 'max_depth': 4, 'min_child_weight': 4, <br>'subsample': 1.0}<br>
# Best Estimator f1-score: 0.9647654766964692<br>

# In[ ]:


# Load trained models
import joblib
best_estimators = joblib.load('best_estimators_dict.joblib')


# In[ ]:


evaluate = ['rf','xgb']
y_test_str = ord_encoders['NObeyesdad'].inverse_transform(np.asarray(y_test).reshape(-1,1))
for mdl in evaluate:
    y_p = ord_encoders['NObeyesdad'].inverse_transform(best_estimators[mdl].predict(X_test).reshape(-1,1))
    print('Classification Report for {}'.format(mdl))
    print(classification_report(y_test_str,y_p))
    print()


# ## XGBoost Classifier Performs Slightly Better than RF
# Examine the confusion matrix for the xgb and rf models.

# In[ ]:


# Confusion Matrix
labels = list(ord_encoders['NObeyesdad'].categories_[0]) # get category labels

# plot heatmap
f,ax = plt.subplots(nrows = 1, ncols = 1, figsize=(10,8))
plot_confusion_matrix(best_estimators['xgb'],X_test,y_test,
                      xticks_rotation='vertical',display_labels=labels,
                      cmap=plt.cm.Greens, ax = ax
                     )
ax.grid(False)
_ = ax.set_title('Confusion Matrix for XGBoost Classifier',
            fontweight='bold')


# In[ ]:


# plot heatmap
f,ax = plt.subplots(nrows = 1, ncols = 1, figsize=(10,8))
plot_confusion_matrix(best_estimators['rf'],X_test,y_test,
                      xticks_rotation='vertical',display_labels=labels,
                      cmap=plt.cm.Greens, ax = ax
                     )
ax.grid(False)
_ = ax.set_title('Confusion Matrix for Random Forest Classifier',
            fontweight='bold')

# Commented out to avoid overwriting models
import joblib
joblib.dump(best_estimators,'best_estimators_dict.joblib')
# In[ ]:




