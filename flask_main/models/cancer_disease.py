import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
import joblib


df=pd.read_csv('../dataset/cancer_data.csv')
print(df.head())

df.drop(["id","Unnamed: 32"],axis=1,inplace=True)

df.rename(columns={'concave points_mean':'concave_points_mean',
                     'concave points_worst':'concave_points_worst'},inplace=True)

df['diagnosis']=df['diagnosis'].map({'M':0,'B':1})
df.head()
#(M = malignant, B = benign)

X=df.drop(['diagnosis'],axis=1)
y=df.diagnosis


from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier()
model.fit(X,y)

feat_importances = pd.Series(model.feature_importances_, index=X.columns)
print(feat_importances.nlargest(12).index)

data_selected=df[['concave_points_worst', 'concave_points_mean', 'perimeter_mean',
       'radius_worst', 'area_worst', 'concavity_worst', 'perimeter_worst',
       'concavity_mean', 'radius_mean', 'compactness_worst', 'area_mean',
       'area_se']]

X=data_selected
y=df.diagnosis


scale=StandardScaler()
X=scale.fit_transform(X)


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=45)
print("Train dataset",X_train.shape,y_train.shape)
print("Test dataset",X_test.shape,y_test.shape)

log_reg=LogisticRegression()

param_grid_log = {'penalty' : ['l2'],
                  'C': [0.1,1.0,10,20,100],
                  'solver' : ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                'max_iter':[10000]
                 }

log_reg_CV = RandomizedSearchCV(estimator=log_reg, param_distributions=param_grid_log, cv=10,
                             n_iter=10,scoring='neg_mean_squared_error',random_state=5,n_jobs=1,verbose=False)

log_reg_CV.fit(X_train, y_train)

joblib.dump(log_reg_CV,"cancer_model")