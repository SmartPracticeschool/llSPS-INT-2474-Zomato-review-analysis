#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic(u'matplotlib inline')

import warnings
warnings.filterwarnings('ignore')


# In[2]:


data = pd.read_csv('/Users/tejaswinikambaiahgari/Downloads/zomato.csv')


# In[3]:


data.head()


# In[4]:


data.isna().sum()


# In[5]:


data=data[data.cuisines.isna()==False]


# In[6]:


data.isna().sum()


# In[7]:


data.drop(columns=["url", 'address','phone','listed_in(city)'], inplace  =True)


# In[8]:


data.rename(columns={'approx_cost(for two people)': 'average_cost'}, inplace=True)


# In[9]:


data['average_cost'].value_counts()[:20]


# In[10]:


dishes_data = data[data.dish_liked.notnull()]
dishes_data.dish_liked = dishes_data.dish_liked.apply(lambda x:x.lower().strip())


# In[11]:


dishes_data.isnull().sum()


# In[12]:


data['rate'] = data['rate'].replace('NEW',np.NaN)
data['rate'] = data['rate'].replace('-',np.NaN)
data.dropna(how = 'any', inplace = True)


# In[13]:


data['rate'] = data.loc[:,'rate'].replace('[ ]','',regex = True)
data['rate'] = data['rate'].astype(str)
data['rate'] = data['rate'].apply(lambda r: r.replace('/5',''))
data['rate'] = data['rate'].apply(lambda r: float(r))


# In[14]:


cuisines_data = data[data.cuisines.notnull()]
cuisines_data.cuisines = cuisines_data.cuisines.apply(lambda x:x.lower().strip())


# In[15]:


data['online_order']= pd.get_dummies(data.online_order, drop_first=True)
data['book_table']= pd.get_dummies(data.book_table, drop_first=True)
data


# In[16]:


data.drop(columns=['dish_liked','reviews_list','menu_item','listed_in(type)'], inplace  =True)


# In[17]:


data['rest_type'] = data['rest_type'].str.replace(',' , '') 
data['rest_type'] = data['rest_type'].astype(str).apply(lambda x: ' '.join(sorted(x.split())))
data['rest_type'].value_counts().head()


# In[18]:


data['cuisines'] = data['cuisines'].str.replace(',' , '') 
data['cuisines'] = data['cuisines'].astype(str).apply(lambda x: ' '.join(sorted(x.split())))
data['cuisines'].value_counts().head()


# In[19]:


from sklearn.preprocessing import LabelEncoder
T = LabelEncoder()                 
data['location'] = T.fit_transform(data['location'])
data['rest_type'] = T.fit_transform(data['rest_type'])
data['cuisines'] = T.fit_transform(data['cuisines'])


# In[20]:


data["average_cost"] = data["average_cost"].str.replace(',' , '') 
data["average_cost"] = data["average_cost"].astype('float')
data.head()


# In[21]:


x = data.drop(['rate','name'],axis = 1)


# In[22]:


y = data['rate']


# In[23]:


x.shape


# In[24]:


y.shape


# In[25]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state = 33)


# In[26]:


from sklearn.preprocessing import StandardScaler
num_values1=data.select_dtypes(['float64','int64']).columns
scaler = StandardScaler()
scaler.fit(data[num_values1])
data[num_values1]=scaler.transform(data[num_values1])


# In[27]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train,y_train)
y_pred_lr = lr.predict(X_test)


# In[28]:


lr.score(X_test, y_test)*100


# In[29]:


from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor()
rfr.fit(X_train,y_train)
y_pred_rfr = rfr.predict(X_test)


# In[30]:


rfr.score(X_test,y_test)*100


# In[31]:


onehot = pd.read_csv("/Users/tejaswinikambaiahgari/Downloads/zomato.csv")
onehot.head()


# In[32]:


onehot['rate'] = onehot['rate'].replace('NEW',np.NaN)
onehot['rate'] = onehot['rate'].replace('-',np.NaN)
onehot.dropna(how = 'any', inplace = True)

onehot['rate'] = onehot.loc[:,'rate'].replace('[ ]','',regex = True)
onehot['rate'] = onehot['rate'].astype(str)
onehot['rate'] = onehot['rate'].apply(lambda r: r.replace('/5',''))
onehot['rate'] = onehot['rate'].apply(lambda r: float(r))


# In[33]:


onehot['cuisines'] = onehot['cuisines'].str.replace(',' , '') 
onehot['cuisines'] = onehot['cuisines'].astype(str).apply(lambda x: ' '.join(sorted(x.split())))
onehot['cuisines'].unique()


# In[34]:


onehot['rest_type'] = onehot['rest_type'].str.replace(',' , '') 
onehot['rest_type'] = onehot['rest_type'].astype(str).apply(lambda x: ' '.join(sorted(x.split())))
onehot['rest_type'].value_counts().head()


# In[35]:


onehot['dish_liked'] = onehot['dish_liked'].str.replace(',' , '') 
onehot['dish_liked'] = onehot['dish_liked'].astype(str).apply(lambda x: ' '.join(sorted(x.split())))
onehot['dish_liked'].value_counts().head()


# In[36]:


dummy_rest_type=pd.get_dummies(onehot['rest_type'])
dummy_city=pd.get_dummies(onehot['location'])
dummy_cuisines=pd.get_dummies(onehot['cuisines'])
dummy_dishliked=pd.get_dummies(onehot['dish_liked'])


# In[37]:


final=pd.concat([onehot,dummy_rest_type,dummy_city,dummy_cuisines,dummy_dishliked],axis=1)


# In[38]:


final.drop(columns=['rest_type','location','cuisines','dish_liked','name','phone'] , inplace=True)
final.drop(columns=['reviews_list','menu_item','listed_in(type)','listed_in(city)'], inplace=True)
final.drop(columns=['url','address'], inplace=True)
final['online_order']=pd.get_dummies(final['online_order'])
final['book_table']=pd.get_dummies(final['book_table'])
final


# In[39]:


final['approx_cost(for two people)'] = final['approx_cost(for two people)'].str.replace(',' , '') 
x = final.drop(['rate'],axis=1)
y = final['rate']


# In[40]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state = 33)


# In[ ]:





# In[41]:


from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor()
rfr.fit(X_train,y_train)
y_pred_rfr = rfr.predict(X_test)
rfr.score(X_test,y_test)*100


# In[ ]:





# In[ ]:





# In[ ]:




