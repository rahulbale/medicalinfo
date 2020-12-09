import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,RandomizedSearchCV
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import resample
from sklearn.linear_model import LogisticRegression
import joblib


df=pd.read_csv('../dataset/diabetes.csv')
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

print(outlier(df,'Glucose'))
print("Total : ", df[df.Glucose == 0].shape[0])

print(outlier(df,'BloodPressure'))
print("Total : ", df[df.BloodPressure == 0].shape[0])
print(df.BloodPressure.mean())

print("Total : ", df[df.SkinThickness == 0].shape[0])
print(df.SkinThickness.mean())

print(outlier(df,'BMI'))
print("Total : ", df[df.BMI == 0].shape[0])

print(extrem_outlier(df,'Insulin'))
print("Total : ", df[df.Insulin == 0].shape[0])
print(df.Insulin.mean())

print(extrem_outlier(df,'DiabetesPedigreeFunction'))
print(df.DiabetesPedigreeFunction.mean())

data=df.copy()

data.loc[data['Glucose']<=37.125,'Glucose']=37
data.loc[data['BloodPressure']<=35.0,'BloodPressure']=69
data.loc[data['SkinThickness']<=0,'SkinThickness']=20
data.loc[data['BMI']<=13.35,'BMI']=13
data.loc[data['Insulin']<=0,'Insulin']=79
data.loc[data['Insulin']>=509.0,'Insulin']=509
data.loc[data['DiabetesPedigreeFunction']>=1.77375,'DiabetesPedigreeFunction']=0.471


data.Outcome.value_counts()
minority=data[data.Outcome==1]
majority=data[data.Outcome==0]

print("minority size",minority.shape)
print("majority size",majority.shape)
min_upsample=resample(minority,replace=True,n_samples=majority.shape[0])
print("minority upsample size",min_upsample.shape)
data=pd.concat([min_upsample,majority],axis=0)
print("After resample",data.Outcome.value_counts())


X=data.drop(['Outcome'],axis=1)
y=data['Outcome']

sc= StandardScaler()
X=sc.fit_transform(X)



X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=45)
print("Train dataset",X_train.shape,y_train.shape)
print("Test dataset",X_test.shape,y_test.shape)


"""
test_scores = []
train_scores = []
for i in range(1, 15):
    knn = KNeighborsClassifier(i)
    knn.fit(X_train, y_train)
    train_scores.append(knn.score(X_train, y_train))
    test_scores.append(knn.score(X_test, y_test))
max_test_score = max(test_scores)
test_scores_ind = [i for i, v in enumerate(test_scores) if v == max_test_score]
print('Max test score {} % and k = {}'.format(max_test_score*100,list(map(lambda x: x+1, test_scores_ind))))
knn = KNeighborsClassifier(3)
knn.fit(X_train,y_train)
print(knn.score(X_test,y_test))
filename='diabeties_modal.pkl'
pickle.dump(knn, open(filename, 'wb'))"""

log_reg=LogisticRegression()

param_grid_log = {'penalty' : ['l2'],
                  'C': [0.1,1.0,10,20,100],
                  'solver' : ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
                 }

log_reg_CV = RandomizedSearchCV(estimator=log_reg, param_distributions=param_grid_log, cv=10,
                             n_iter=10,scoring='neg_mean_squared_error',random_state=5,n_jobs=1,verbose=False)

log_reg_CV.fit(X_train, y_train)


joblib.dump(log_reg_CV,"diabetes_model")