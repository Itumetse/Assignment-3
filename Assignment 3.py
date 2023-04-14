#!/usr/bin/env python
# coding: utf-8

# # COUNTRY-WISE COVID CASES <br/>
#  
# **AUTHOR:** *SUNAYANA GAWDE* <br/>
# **DATE:** *14 April 2023* <br/>
# **SOURCE:** *Kaggle*<br/>
# **GITHUB URL:**  

# In[18]:


#Import Libraries

import pandas as pd
import numpy as np
from pandas import Series, DataFrame

import scipy
from scipy import stats

data = pd.read_excel("Country-wise-COVID-cases_v5.xlsx")


# In[4]:


data.head()


# ## DESCRIPTIVE STATISTICS

# In[20]:


Data = pd.read_excel("Country-wise-COVID-cases_v5.xlsx", usecols = [1,2,3,4,5])
Data


# In[21]:


Data.sum()


# In[22]:


Data.mean()


# In[23]:


Data.median()


# In[24]:


Data.max()


# In[38]:


Data.min()


# In[25]:


Data.mode()


# In[26]:


Data.std()


# In[27]:


Data.var()


# In[32]:


from scipy.stats import gmean
Geomean = gmean(Data)
print("Geometric Mean: ", Geomean)


# ### DESCRIPTIVE STATISTICS - DATA 1

# In[42]:


Data1 = pd.read_excel("Country-wise-COVID-cases_v5.xlsx", usecols = [1])
Data1


# In[48]:


#Mean
Data1.mean()


# In[49]:


#Mode
Data1.mode()


# In[50]:


#Median
Data1.median()


# In[53]:


#Maximum 
Data1.max()


# In[54]:


#Minimum
Data1.min()


# In[52]:


#Range
Range = Data1.max() - Data1.min()
print(Range)


# In[55]:


#Geometric mean
from pandas import DataFrame
from scipy.stats.mstats import gmean

df = DataFrame(Data1)

geometric_mean = gmean(df)
print(geometric_mean)


# In[59]:


#Percentiles
Q1 = np.percentile(Data1, 25)
print(Q1)


# In[60]:


Q2 = np.percentile(Data1, 50)
print(Q2)


# In[61]:


Q3 = np.percentile(Data1, 75)
print(Q3)


# In[47]:


Q1 = np.percentile(Data1, 25)
Q3 = np.percentile(Data1, 75)
iqr = Q3 - Q1
print (iqr)


# In[56]:


#Variance
Data1.var()


# In[57]:


#Standard Deviation
Data1.std()


# In[58]:


#Five Number Summary
Data1.describe()


# In[62]:


#Coefficient of Variation
CV = Data1.std()/Data1.mean()
print(CV)


# In[63]:


#Histogram

import matplotlib.pyplot as plt

x = np.random.normal(Data1)

plt.hist(x)
plt.show()


# In[68]:


#Z-Scores

import pandas as pd
import numpy as np
import scipy.stats as stats

x = np.array(Data1)

stats.zscore(x)


# In[90]:


Data2 = pd.read_excel("Country-wise-COVID-cases_v5.xlsx", usecols = [2])
Data2


# In[91]:


#CoVariance
CV = np.array(Data)
cov_matrix = np.cov(CV, bias=True)
print(cov_matrix)


# In[93]:


#correlation efficient 
import seaborn as sn
import matplotlib.pyplot as plt

sn.heatmap(cov_matrix, annot=True, fmt="g")
plt.show()


# In[94]:


cov_matrix = np.cov(CV, bias=False)
print(cov_matrix)


# In[104]:


#Histogram

import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')

Mean = Data1.mean()
Stdev = Data1.std()
size = 212
var = np.random.normal(Mean, Stdev, size)

plt.hist(var)
plt.show()


# In[ ]:


#correlation coeffiecient

import numpy as np
import matplotlib.pyplot as plt

x = np.array([Data1])
y = np.array([Data2])
ans = np.corrcoef(x,y)
ans


# In[ ]:


#BoxPlot
fig = Data.boxplot(size = 212)
fig.show()


# In[ ]:


#Scatter Plot
fig = Data.scatter(x = Data, y = Data1)
fig.show()

