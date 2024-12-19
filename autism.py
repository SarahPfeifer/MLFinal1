from ucimlrepo import fetch_ucirepo 
import scipy as sp
import scipy.stats as stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import copy
# Set color map to have light blue background
sns.set()
import statsmodels.formula.api as smf
import statsmodels.api as sm
 
# fetch dataset 
autism_screening_adult = fetch_ucirepo(id=426) 
  
# data (as pandas dataframes) 
X = autism_screening_adult.data.features 
y = autism_screening_adult.data.targets 
  
# metadata 
print(autism_screening_adult.metadata) 
  
# variable information
print(autism_screening_adult.variables) 
X.jaundice.replace(('yes', 'no'), (1, 0), inplace=True)
X.family_pdd.replace(('yes', 'no'), (1, 0), inplace=True)
X.gender.replace(('m', 'f'), (1,0), inplace=True)
X.age.replace(383, 38, inplace=True)
X.ethnicity.replace('others', 'Others', inplace=True)
X['ethnicity'].fillna('Others', inplace=True)
X.drop(['age_desc', 'relation', 'used_app_before'], axis=1, inplace=True)
mean_value = int(X['age'].mean())
X['age'].fillna(value=mean_value, inplace=True) 
for c in X.columns[:]:
    print(c, X[c].unique())
X['age'].fillna

#Encode Categorical Data
ohe = OneHotEncoder()
encoded_data = ohe.fit_transform(X[['ethnicity']])
X_encoded = pd.DataFrame(encoded_data.toarray(), columns=ohe.get_feature_names_out(['ethnicity']))
X = pd.concat([X.drop('ethnicity', axis=1), X_encoded], axis=1)
encoded_data = ohe.fit_transform(X[['country_of_res']])
X_encoded = pd.DataFrame(encoded_data.toarray(), columns=ohe.get_feature_names_out(['country_of_res']))
X = pd.concat([X.drop('country_of_res', axis=1), X_encoded], axis=1)

#Split Dataset

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Random Forest Baseline

rfc = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train,y_train)
yhat = rfc.predict(X_test)
accuracy = accuracy_score(y_test, yhat)
print('Accuracy:', accuracy)