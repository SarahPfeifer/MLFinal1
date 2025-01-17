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
from dfcorrs.cramersvcorr import Cramers

 
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

#Create subplots
fig, axs = plt.subplots(2, 5, figsize=(10,5))
j=0
k=0
# Plot pie charts
for i, col in enumerate(X.columns[:10]):
    counts = X[col].value_counts()
    axs[j,k].pie(counts, labels=counts.index)
    axs[j,k].set_title(col)
    if k < 4:
        k+=1
    elif j < 1:
        k=0
        j+=1
plt.tight_layout()
fig.suptitle('AQ 10 Responses by Question')    
plt.show()

# Create the figure and axes
fig, axs = plt.subplots(1, 3, figsize=(5,3))

# First row (3 subplots)
counts = X[X.columns[11]].value_counts()
axs[0].pie(counts, labels=counts.index)
axs[0].set_title(X.columns[11])
counts = X[X.columns[13]].value_counts()
axs[1].pie(counts, labels=counts.index)
axs[1].set_title(X.columns[13])
counts = X[X.columns[14]].value_counts()
axs[2].pie(counts, labels=counts.index)
axs[2].set_title(X.columns[14])

fig.suptitle('Responses for Binary Demographic Features')   
# Adjust spacing between subplots
plt.tight_layout()

# Show the plot
plt.show()

# Create the figure and axes
fig, axs = plt.subplots(1, 4, figsize=(10,3))

# Second row (4 subplots)
counts = X[X.columns[10]].value_counts()
axs[0].pie(counts)
axs[0].set_title(X.columns[10])
counts = X[X.columns[12]].value_counts()
axs[1].pie(counts)
axs[1].set_title(X.columns[12])
counts = X[X.columns[15]].value_counts()
axs[2].pie(counts)
axs[2].set_title(X.columns[15])
counts = X[X.columns[16]].value_counts()
axs[3].pie(counts)
axs[3].set_title(X.columns[16])

fig.suptitle('Responses for Categorical Demographic Features') 
# Adjust spacing between subplots
plt.tight_layout()

# Show the plot
plt.show()

#Encode Categorical Data
ohe = OneHotEncoder()
encoded_data = ohe.fit_transform(X[['ethnicity']])
X_encoded = pd.DataFrame(encoded_data.toarray(), columns=ohe.get_feature_names_out(['ethnicity']))
X = pd.concat([X.drop('ethnicity', axis=1), X_encoded], axis=1)
encoded_data = ohe.fit_transform(X[['country_of_res']])
X_encoded = pd.DataFrame(encoded_data.toarray(), columns=ohe.get_feature_names_out(['country_of_res']))
X = pd.concat([X.drop('country_of_res', axis=1), X_encoded], axis=1)

#Check for Imbalance, Target Data
target_counts = y['class'].value_counts()

#Create Visualization
target_counts.plot(kind='bar')
plt.title('Target Variable Class Distribution')
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.show()

#Check for Imbalance, Features

# Create subplots




#Split Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Random Forest Baseline
n_estimators=list(range(1,21))
accuracy_list=[]
for e in n_estimators:
    rfc = RandomForestClassifier(n_estimators=e, random_state=42).fit(X_train,y_train)
    yhat = rfc.predict(X_test)
    accuracy = accuracy_score(y_test, yhat)
    accuracy_list.append(accuracy)
#print('Accuracy:', accuracy)

fig, ax = plt.subplots()
ax.set_xlabel("Number of Trees")
ax.set_ylabel("Accuracy")
ax.set_title("Accuracy vs Number of Trees: Autism Prediction")
ax.plot(np.arange(0,len(n_estimators)), accuracy_list, marker=',',
        drawstyle="steps-post")
plt.show() 
