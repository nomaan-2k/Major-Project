# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 23:32:39 2022

@author: nomaa
"""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.feature_selection
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import RocCurveDisplay
import sklearn.metrics as met

df = pd.read_csv('diabetes.csv')
print(df.head(5))

X=df.drop('Outcome',1)
y=df.Outcome



print(X.head(5))
print(y.head(5))



X.Glucose = X.Glucose.replace(0, np.nan)
X.BloodPressure = X.BloodPressure.replace(0, np.nan)
X.SkinThickness = X.SkinThickness.replace(0, np.nan)
X.Insulin = X.Insulin.replace(0, np.nan)
X.BMI = X.BMI.replace(0, np.nan)
X.Age = X.Age.replace(0, np.nan)

# How much of your data is missing?
print(X.isna().sum().sort_values(ascending=False).head())

# Impute missing values using Imputer in sklearn.preprocessing


imp = SimpleImputer(missing_values=np.nan, strategy='median')
imp.fit(X)
X = pd.DataFrame(data=imp.transform(X) , columns=X.columns)
print(X.head(5))

# Now check again to see if you still have missing data
print(X.isna().sum().sort_values(ascending=False).head())


# Use train_test_split in sklearn.cross_validation to split data into train and test sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.70, random_state=1)
# Function to build model and find model performance
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=270)
model.fit(X_train, y_train)

feature_imp = pd.Series(model.feature_importances_,index=X.columns).sort_values(ascending=False)
print("Feature Importance :\n\n",feature_imp)


# Creating a bar plot
sns.barplot(x=feature_imp, y=feature_imp.index)
# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.legend()
plt.show()

# The total number of features have grown substantially after dummying and adding interaction terms
print(df.shape)
print(X.shape)


select = sklearn.feature_selection.SelectKBest(k=5)
selected_features = select.fit(X_train, y_train)
indices_selected = selected_features.get_support(indices=True)
colnames_selected = [X.columns[i] for i in indices_selected]

X_train_selected = X_train[colnames_selected]
X_test_selected = X_test[colnames_selected]


print(colnames_selected)




# parameters ={
#     "n_estimators":[5,10,50,100,250,300,400],
#     "max_depth":[2,4,8,16,32,None]
#     }

# cv=GridSearchCV(RandomForestClassifier(),parameters,cv=5)
# cv.fit(X_train_selected,y_train.values.ravel())

# def display(results):
#     print(f'Best parameters are: {results.best_params_}')
#     print("\n")
#     mean_score = results.cv_results_['mean_test_score']
#     std_score = results.cv_results_['std_test_score']
#     params = results.cv_results_['params']
#     for mean,std,params in zip(mean_score,std_score,params):
#         print(f'{round(mean,3)} + or -{round(std,3)} for the {params}')
        
# display(cv)



def find_model_perf(X_train, y_train, X_test, y_test):
    model = RandomForestClassifier(n_estimators=270)
    model.fit(X_train, y_train)
    y_pred=model.predict(X_test)
    y_hat = [x[1] for x in model.predict_proba(X_test)]
    RocCurveDisplay.from_predictions(y_test, y_hat)
    print("Accuracy : ",met.roc_auc_score(y_test,y_hat))
    print("Log Loss : ",sklearn.metrics.log_loss(y_test, y_pred)) 
    print("Precision:",met.precision_score(y_test, y_pred))
    print("Recall:",met.recall_score(y_test, y_pred))
    auc = roc_auc_score(y_test, y_hat)
    
    return auc

# Find performance of model using preprocessed data
auc_processed = find_model_perf(X_train_selected, y_train, X_test_selected, y_test)



