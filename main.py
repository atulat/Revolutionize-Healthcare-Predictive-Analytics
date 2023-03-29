
import numpy as np # linear algebra
import pandas as pd # data processing

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

train = pd.read_csv('../input/stoke-prediction-dataset/train.csv')
test = pd.read_csv('../input/stoke-prediction-dataset/test.csv')

train.head()

test.head()

x_train = train.drop(['id','stroke'], axis=1)
y_train = train['stroke']
x_test_id = test['id']
x_test = test.drop('id', axis=1)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)

cols = x_train.select_dtypes(include='object')
for c in cols:
    print(c, x_train[c].unique())

cols = x_test.select_dtypes(include='object')
for c in cols:
    print(c, x_test[c].unique())

x_train.groupby('gender').mean()


i = x_train.loc[x_train.gender=='Other'].index
x_train = x_train.drop(i, axis=0)
y_train = y_train.drop(i, axis=0)

x_train.info()

y_train.shape

x_train['age'] = x_train.age.str.replace('*','', regex=False).astype('int')

x_train.info()

x_train.describe()

x_train.isnull().sum()

cols = ['hypertension','heart_disease','avg_glucose_level']
from sklearn.preprocessing import StandardScaler, MinMaxScaler
ss = StandardScaler()
ss.fit(x_train[cols])
x_train[cols] = ss.transform(x_train[cols])
x_test[cols] = ss.transform(x_test[cols])

objs = x_train.select_dtypes(include='object').columns
x_train = pd.get_dummies(x_train, columns = objs)
x_test = pd.get_dummies(x_test, columns = objs)

## bmi 
bmi_x_tr = x_train.loc[x_train.bmi.notnull(),:].drop('bmi',axis=1)
bmi_x_tt = x_test.loc[x_test.bmi.notnull(),:].drop('bmi',axis=1)
bmi_tr = x_train.loc[x_train.bmi.notnull(),'bmi']
bmi_tt = x_test.loc[x_test.bmi.notnull(),'bmi']

bmi_x_tr_null = x_train.loc[x_train.bmi.isnull(),:].drop('bmi', axis=1)
bmi_x_tt_null = x_test.loc[x_test.bmi.isnull(),:].drop('bmi',axis=1)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(bmi_x_tr, bmi_tr)
bmi_tr_pred = lr.predict(bmi_x_tr_null)
bmi_tt_pred = lr.predict(bmi_x_tt_null)

x_train.loc[x_train.bmi.isnull(),'bmi'] = bmi_tr_pred
x_test.loc[x_test.bmi.isnull(),'bmi'] = bmi_tt_pred



# 다시 bmi를 scaling
ss.fit(x_train[['bmi']])
x_train['bmi'] = ss.transform(x_train[['bmi']])
x_test['bmi'] = ss.transform(x_test[['bmi']])

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, f1_score, recall_score, precision_score

from sklearn.model_selection import train_test_split
x_tr, x_val, y_tr, y_val = train_test_split(x_train, y_train, test_size = 0.3, random_state=2022)

rf = RandomForestClassifier(n_estimators = 300, max_depth = 5, min_samples_split = 5, random_state=2022)
rf.fit(x_tr, y_tr)
pred = rf.predict(x_val)
pred_proba = rf.predict_proba(x_val)
accuracy = accuracy_score(y_val, pred)
roc_auc = roc_auc_score(y_val, pred_proba[:,1])

print('accuracy', accuracy)
print('roc_auc', roc_auc)

xg = XGBClassifier(n_estimators = 300, max_depth = 5, learning_rate = 0.08, random_state=2022)
xg.fit(x_tr, y_tr)
pred = xg.predict(x_val)
pred_proba = xg.predict_proba(x_val)
accuracy = accuracy_score(y_val, pred)
roc_auc = roc_auc_score(y_val, pred_proba[:,1])

print('accuracy', accuracy)
print('roc_auc', roc_auc)

## RandomForest
rf = RandomForestClassifier(n_estimators = 300, max_depth = 5, min_samples_split = 5, random_state=2022)
rf.fit(x_train, y_train)
pred = rf.predict(x_test)
pred_proba = rf.predict_proba(x_test)
pd.DataFrame({'id' : x_test_id, 'predict' : pred_proba[:,1]}).to_csv('00000000.csv', index=False)

