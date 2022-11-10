#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression


# In[2]:


ad_data = pd.read_csv(r'C:\Users\sreedev\Downloads\advertising_data.csv')
ad_data.head()


# In[3]:


ad_data.info()


# In[4]:


ad_data.describe(include = 'all')


# In[ ]:





# ## EDA

# In[5]:


sns.jointplot(ad_data['Age'],ad_data['Area Income'])


# In[6]:


sns.jointplot(ad_data['Daily Time Spent on Site'],ad_data['Daily Internet Usage'])


# In[7]:


sns.pairplot(ad_data, hue = 'Clicked on Ad')


# In[ ]:





# ## Regression Model

# In[8]:


y = ad_data['Clicked on Ad']
x = ad_data[['Daily Time Spent on Site','Age','Area Income','Daily Internet Usage','Male']]


# In[9]:


from sklearn.model_selection import  train_test_split


# In[10]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.20,random_state = 42)


# In[11]:


LR = LogisticRegression()


# In[12]:


LR.fit(x_train,y_train)


# In[13]:


predictions = LR.predict(x_test)


# In[14]:


reg_log = sm.Logit(y,x)
results_log = reg_log.fit()
results_log.summary()


# In[15]:


results_log.pred_table()


# In[16]:


cm_df = pd.DataFrame(results_log.pred_table())
cm_df.columns = ['Predicted 0', 'Predicted 1']
cm_df


# In[17]:


from sklearn.metrics import classification_report


# In[18]:


print(classification_report(y_test,predictions))

