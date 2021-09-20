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


# In[2]:


pip install dataeader


# In[3]:


import datareader


# In[4]:


data = pd.read_csv('C:\Files for python projects\Tesla.csv')


# In[5]:


data.head()


# In[6]:


data.describe


# In[7]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (15, 10)


# In[8]:


data['Date'] = pd.DatetimeIndex(data['Date']).year
# adding a new year into our dataset


# In[9]:


tempdf = data.groupby('Date',as_index=False).sum()
tempdf.plot('Date',['Open','Close','High','Low'],kind = 'bar')
# how our data looks from troughout 7 years


# In[10]:


x = data.groupby('Date').sum().sort_values('Volume', ascending = False)
x.plot.bar( y='Volume', rot=0)


# In[11]:


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


# In[ ]:


Graphical visualisation within the set years


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





# In[17]:


data['PriceDiff'] = data['Close'].shift(-1) - data['Close']
print(data['PriceDiff'])


# In[18]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# In[19]:


data['PriceDiff'] = data['Close'].shift(-1) - data['Close']
print(data['PriceDiff'])


# In[20]:


data['ma50'] = data['Close'].rolling(50).mean()
#plot the moving average
plt.figure(figsize=(10, 8))
data['ma50'].plot(label='MA50')
data['Close'].plot(label='Close')
plt.legend()
plt.show()


# In[21]:


data['LogReturn'] = np.log(data['Close']).shift(-1) - np.log(data['Close'])
print(data['LogReturn'])


# In[22]:


data['MA10'] = data['Close'].rolling(10).mean()
data['MA50'] = data['Close'].rolling(50).mean()
data = data.dropna()
data.head()


# In[23]:


data['Shares'] = [1 if data.loc[ei, 'MA10']>data.loc[ei, 'MA50'] else 0 for ei in data.index]


# In[24]:


data['Close1'] = data['Close'].shift(-1)
data['Profit'] = [data.loc[ei, 'Close1'] - data.loc[ei, 'Close'] if data.loc[ei, 'Shares']==1 else 0 for ei in data.index]
data['Profit'].plot()
plt.axhline(y=0, color='red')


# In[25]:


data['wealth'] = data['Profit'].cumsum()
data.tail()


# In[26]:


pip install pandas_datareader


# In[27]:


from pandas_datareader import data


# In[38]:


data = data.DataReader('TSLA', 'yahoo',start='1/1/2000')


# In[29]:


time_elapsed = (data.index[-1] - data.index[0]).days


# In[30]:


price_ratio = (data['Adj Close'][-1] / data['Adj Close'][1])
inverse_number_of_years = 365.0 / time_elapsed
cagr = price_ratio ** inverse_number_of_years - 1
print(cagr)


# In[31]:


vol = data['Adj Close'].pct_change().std()


# In[32]:


number_of_trading_days = 252
vol = vol * math.sqrt(number_of_trading_days)


# In[33]:


print ("cagr (mean returns) : ", str(round(cagr,4)))
print ("vol (standard deviation of return : )", str(round(vol,4)))


# In[34]:


daily_return_percentages = np.random.normal(cagr/number_of_trading_days, vol/math.sqrt(number_of_trading_days),number_of_trading_days)+1


# In[35]:


price_series = [data['Adj Close'][-1]]

for drp in daily_return_percentages:
    price_series.append(price_series[-1] * drp)


# In[36]:


plt.plot(price_series)
plt.show()


# In[37]:


number_of_trials = 10000
for i in range(number_of_trials):
    daily_return_percentages = np.random.normal(cagr/number_of_trading_days, vol/math.sqrt(number_of_trading_days),number_of_trading_days)+1
    price_series = [data['Adj Close'][-1]]

    for drp in daily_return_percentages:
        price_series.append(price_series[-1] * drp)
    
    plt.plot(price_series)
plt.show()


# In[38]:


ending_price_points = []
larger_number_of_trials = 9001 
for i in range(larger_number_of_trials):
    daily_return_percentages = np.random.normal(cagr/number_of_trading_days, vol/math.sqrt(number_of_trading_days),number_of_trading_days)+1
    price_series = [data['Adj Close'][-1]]

    for drp in daily_return_percentages:
        price_series.append(price_series[-1] * drp)
    
    plt.plot(price_series)
    
    ending_price_points.append(price_series[-1])

plt.show()

plt.hist(ending_price_points,bins=50)
plt.show()


# In[39]:


expected_ending_price_point = round(np.mean(ending_price_points),2)
print("Expected Ending Price Point : ", str(expected_ending_price_point))


# In[40]:


population_mean = (cagr+1) * data['Adj Close'][-1]
print ("Sample Mean : ", str(expected_ending_price_point))
print ("Population Mean: ", str(round(population_mean,2)));
print ("Percent Difference : ", str(round((population_mean - expected_ending_price_point)/population_mean * 100,2)), "%")


# In[41]:


top_ten = np.percentile(ending_price_points,100-10)
bottom_ten = np.percentile(ending_price_points,10);
print ("Top 10% : ", str(round(top_ten,2)))
print ("Bottom 10% : ", str(round(bottom_ten,2)))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




