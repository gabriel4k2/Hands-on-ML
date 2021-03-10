#!/usr/bin/env python
# coding: utf-8

# In[23]:


import os
import tarfile
import urllib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.impute import SimpleImputer

from pandas.plotting import scatter_matrix

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing(housing_url=HOUSING_URL, housing_path=HOUSING_PATH ):
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()
    
def load_housing_data( housing_path=HOUSING_PATH ):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)
'''
def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    np.random.seed(42)
    test_size = int(len(data)*test_ratio)
    test_indeces = shuffled_indices[:test_size]
    train_indices = shuffled_indices[test_size:]
    return data.iloc[train_indices], data.iloc[test_indeces]
'''


# In[2]:


fetch_housing()


# In[3]:


housing = load_housing_data()
housing.head()


# In[4]:


housing.info()


# In[5]:


housing.describe()


# In[8]:


get_ipython().run_line_magic('matplotlib', 'inline')
housing.hist(bins=50, figsize=(15,10))
plt.show()


# In[9]:


train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)


# In[10]:


train_set.info()


# In[11]:


housing["income_category"] = pd.cut(housing["median_income"],
                                   bins=[ 0.,1.5,3.0,4.5, 6., np.inf],
                                   labels=[1,2,3,4,5])
housing["income_category"].describe()


# In[12]:


housing["income_category"].hist()


# In[13]:


split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_category"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

strat_test_set["income_category"].value_counts()/ len(strat_test_set)


# In[14]:


#Return the set to the default (income_category as just to get stratified data sets)
#for set_ in (strat_train_set,strat_test_set ):
#    set_.drop("income_category", axis=1,inplace=True)
    
housing = strat_train_set.copy()
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)


# In[15]:


#radius is set by the "s" arg while "c" is the field which will be used to set the color
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
            s=housing['population']/100, label="pop", figsize=(12,9),
            c="median_house_value", cmap=plt.get_cmap("jet"),
            colorbar=True)
plt.legend()


# In[16]:


corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)


# In[17]:


#getting the correlations
attributes = ['median_house_value', 'median_income', 'total_rooms', 'housing_median_age']
scatter_matrix(housing[attributes], figsize=(10,6))


# In[18]:


#median_house_value x median_income seems promising:
housing.plot(kind="scatter", x="median_income", y='median_house_value', alpha=0.1)


# In[19]:


housing['rooms_per_household'] = housing['total_rooms']/housing['households']
housing['bedrooms_per_room'] = housing['total_bedrooms']/housing['total_rooms']
housing['population_per_household'] = housing['population']/housing['households']


# In[20]:


corr_matrix = housing.corr()
corr_matrix['median_house_value'].sort_values(ascending=False)


# In[21]:


housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()


# In[24]:


#Non-numeric
imputer = SimpleImputer(strategy='median')
housing_num =housing.drop("ocean_proximity", axis=1)

imputer.fit(housing_num)


# In[25]:


imputer.statistics_


# In[26]:


#all value that were missing will be swapped with the respective median. IE missing bedroom number field in a specific tuple
#will be swapped to the mean bedroom number. 
# Notice that the object returned is a simple numpy array
X = imputer.transform(housing_num)


# In[29]:


housing_tr = pd.DataFrame(X, columns=housing_num.columns,
                         index=housing_num.index)


# In[31]:


housing_tr.info()


# In[ ]:




