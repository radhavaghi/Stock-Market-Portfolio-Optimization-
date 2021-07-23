#!/usr/bin/env python
# coding: utf-8

# In[1]:


#This program attempts to optimize a user's portfolio using efficient frontier


# In[2]:



pip install pandas_datareader


# In[3]:


#importing python libraries
from pandas_datareader import data as web
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


# In[4]:


#getting stock symbols/tickers in the portfolio
#FAANG- Facebook, Amazon, Apple, Netflix, Google
assets = ['FB', 'AMZN', 'AAPL', 'NFLX', 'GOOG']


# In[5]:


#ASSIGN WEIGHTS TO PORTFOLIO: percentage of each company in the portfolio, must be equal to 1.
#Hypothetically assigning them to 0.2 so each weight is equal for each stock. 
weights= np.array ([0.2, 0.2, 0.2, 0.2, 0.2])



# In[6]:


#GET THE STOCK STARTING DATE
#2013 because FB boomed in that specific year, entirely.
stockstartdate= '2013-01-01'


# In[7]:


#GET STOCKS ENDING DATE (TODAY)
today= datetime.today().strftime('%Y-%m-%d')
today


# In[8]:


#CREATE A DATAFRAME TO STORE THE ADJUSTED CLOSING PRICE OF THE STOCK
df= pd.DataFrame()

#STORE THE ADJUSTED CLOSE PRICE OF THE STOCK INTO THE df
for stock in assets:
    df[stock]= web.DataReader(stock, data_source= 'yahoo', start=stockstartdate, end=today)['Adj Close']
    


# In[9]:


#SHOW THE DATAFRAME
df


# In[10]:


#VISUALLY SHPW THE STOCK PORTFOLIO
title= 'Portfolio Adj.Close Price History'

#GET THE STOCKS
my_stocks=df

#CREATE AND PLOT THE GRAPH
for c in my_stocks.columns.values:
    plt.plot(my_stocks[c], label=c)
    
plt.title(title)
plt.xlabel('Date', fontsize= 18)
plt.ylabel('Adj. Price USD ($)', fontsize= 18)
plt.legend (my_stocks.columns.values, loc= 'upper left')
plt.show()


# In[11]:


#SHOW THE DAILY SIMPLE RETURNS
returns= df.pct_change()
returns


# In[12]:


#SHOW ANNUALIZED COVARAIANCE MATRIX
cov_matrix_annual= returns.cov() * 252
cov_matrix_annual


# In[13]:


#CALCULATE THE PORTFOLIO VARIANCE
port_variance = np.dot(weights.T, np.dot(cov_matrix_annual, weights))
port_variance


# In[14]:


#CALCUATE THE PORTFOLIO VOLATILITY- STANDARD DEVIATION
port_volatility= np.sqrt(port_variance)
port_volatility


# In[15]:


#ANNUAL PORTFOLIO RETURN CALCULATION
portfolioSimpleAnnualReturn = np.sum(returns.mean() * weights) * 252
portfolioSimpleAnnualReturn


# In[16]:


#SHOW EXPECTED ANNUAL RETURN, VOLATILITY (RISK), VARIANCE

percent_var = str(round (port_variance, 2 ) * 100) + '%'
percent_vols = str (round(port_volatility, 2)* 100) + '%'
percent_ret= str(round (portfolioSimpleAnnualReturn, 2)* 100) + '%'

print ('Expected Annual return :'+ percent_ret)
print ('annual volatility or risk :'+ percent_vols)
print('Annual variance :'+ percent_var)


# In[17]:


pip install PyPortfolioOpt


# In[18]:



pip install cvxopt


# In[19]:


from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns


# In[20]:


#PORTFOLIO OPTIMIZATION
#calculating expected returns and annualized sample covaraince matrix of asset returns
mu= expected_returns.mean_historical_return(df)
S= risk_models.sample_cov(df)

#MAXIMAL SHARPE RATIO; a way to describe how much excess return you receive out of volatility. 

ef= EfficientFrontier (mu, S)
weights= ef.max_sharpe()
cleaned_weights= ef.clean_weights()
print(cleaned_weights)
ef.portfolio_performance(verbose = True)


# In[21]:


#GET THE DISCRETE ALLOCATION OF EACH SHARE PER STOCK

from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

latest_prices= get_latest_prices(df)
weights= cleaned_weights
da = DiscreteAllocation(weights, latest_prices, total_portfolio_value = 15000 )
allocation, leftover= da.lp_portfolio()

print('Discrete allocation:', allocation)

print('Funds remaining: $ {:.2f}'.format(leftover))


# In[ ]:





# In[ ]:




