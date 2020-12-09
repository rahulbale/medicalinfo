import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
import joblib


df=pd.read_csv('../dataset/heart.csv')
print(df.head())

def outlier(df,variable):
    IQR=df[variable].quantile(0.75)-df[variable].quantile(0.25)
    lower_bound=df[variable].quantile(0.25)-(IQR*1.5)
    upper_bound=df[variable].quantile(0.75)+(IQR*1.5)
    return lower_bound,upper_bound

def extrem_outlier(df,variable):
    IQR=df[variable].quantile(0.75)-df[variable].quantile(0.25)
    lower_bound=df[variable].quantile(0.25)-(IQR*3)
    upper_bound=df[variable].quantile(0.75)+(IQR*3)
    return lower_bound,upper_bound

outlier(df,'trestbps')
extrem_outlier(df,'chol')
outlier(df,'thalach')


data=df.copy()

data.loc[data['trestbps']>=170.0,'trestbps']=170
data.loc[data['chol']>=465.0,'chol']=465
data.loc[data['thalach']<=84.75,'thalach']=84


X=data.drop(['target'],axis=1)
y=data['target']


scale=StandardScaler()
X=scale.fit_transform(X)
print(X)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=45)
print("Train dataset",X_train.shape,y_train.shape)
print("Test dataset",X_test.shape,y_test.shape)


param_grid_log = {'penalty' : ['l2'],
                  'C': [0.1,1.0,10,20,100],
                  'solver' : ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
                 }
log_reg=LogisticRegression()

log_reg_CV = RandomizedSearchCV(estimator=log_reg, param_distributions=param_grid_log, cv=10,
                             n_iter=10,scoring='neg_mean_squared_error',random_state=5,n_jobs=1,verbose=False)

log_reg_CV.fit(X_train, y_train)


joblib.dump(log_reg_CV,"heart_model")


