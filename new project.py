#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")
get_ipython().run_line_magic('matplotlib', 'inline')


# For time stamps
from datetime import datetime


# In[3]:


pip install dataeader


# In[2]:


import datareader


# In[3]:


data = pd.read_csv('C:\Files for python projects\Tesla.csv')


# In[4]:


data.head()


# In[5]:


data.describe


# In[16]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (15, 10)


# In[7]:


data['Date'] = pd.DatetimeIndex(data['Date']).year
# adding a new year into our dataset


# In[8]:


tempdf = data.groupby('Date',as_index=False).sum()
tempdf.plot('Date',['Open','Close','High','Low'],kind = 'bar')
# how our data looks from troughout 7 years


# In[9]:


x = data.groupby('Date').sum().sort_values('Volume', ascending = False)
x.plot.bar( y='Volume', rot=0)


# In[10]:


data.groupby('Date')['Date'].agg('count').plot(kind='pie',title='Date')


# In[12]:


newDf = data[(data.Date==2013)]
newDf.Volume.sum()
s = data.Date.value_counts()
x = s.to_dict()
x


# In[13]:


def yearlyAllocatedData(years_dict,totalSize):
    allocation = []
    for i in years_dict:
        allocation.append((100 * years_dict[i] / totalSize))
    return allocation


# In[14]:


print(yearlyAllocatedData(x,len(data)))


# In[15]:


data = data[data['Date'] != 2013]
data


# In[67]:


Graphical visualisation of opens, highs, closes, lows throught the years


# In[16]:


fig = plt.figure(figsize=(25,20))
plt.subplot(2,2,1)
plt.title('Open')
plt.xlabel('Days')
plt.ylabel('Opening Price USD ($)')
plt.plot(data['Open'])

plt.subplot(2,2,2)
plt.title('Close Price')
plt.xlabel('Days')
plt.ylabel('Closing Price USD ($)')
plt.plot(data['Close'])

plt.subplot(2,2,3)
plt.title('High Price')
plt.xlabel('Days')
plt.ylabel('High Price USD ($)')
plt.plot(data['High'])

plt.subplot(2,2,4)
plt.title('Low Price')
plt.xlabel('Days')
plt.ylabel('Low Price USD ($)')
plt.plot(data['Low'])

plt.show()
# time series analysis


# In[ ]:


# Now let us calcuate Tesla's discounted cash flow to value stock


# In[22]:


data['PriceDiff'] = data['Close'].shift(-1) - data['Close']
print(data['PriceDiff'])


# In[21]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# In[23]:


data['PriceDiff'] = data['Close'].shift(-1) - data['Close']
print(data['PriceDiff'])


# In[25]:


data['ma50'] = data['Close'].rolling(50).mean()
#plot the moving average
plt.figure(figsize=(10, 8))
data['ma50'].plot(label='MA50')
data['Close'].plot(label='Close')
plt.legend()
plt.show()


# In[27]:


data['LogReturn'] = np.log(data['Close']).shift(-1) - np.log(data['Close'])
print(data['LogReturn'])


# In[29]:


data['MA10'] = data['Close'].rolling(10).mean()
data['MA50'] = data['Close'].rolling(50).mean()
data = data.dropna()
data.head()
we added new collumns to perfom further analysis


# In[31]:


data['Shares'] = [1 if data.loc[ei, 'MA10']>data.loc[ei, 'MA50'] else 0 for ei in data.index]


# In[32]:


data['Close1'] = data['Close'].shift(-1)
data['Profit'] = [data.loc[ei, 'Close1'] - data.loc[ei, 'Close'] if data.loc[ei, 'Shares']==1 else 0 for ei in data.index]
data['Profit'].plot()
plt.axhline(y=0, color='red')


# In[33]:


data['wealth'] = data['Profit'].cumsum()
data.tail()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




