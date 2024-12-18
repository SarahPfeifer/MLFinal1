pip install ucimlrepo

from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
autism_screening_adult = fetch_ucirepo(id=426) 
  
# data (as pandas dataframes) 
X = autism_screening_adult.data.features 
y = autism_screening_adult.data.targets 
  
# metadata 
print(autism_screening_adult.metadata) 
  
# variable information 
print(autism_screening_adult.variables) 
