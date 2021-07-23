# Stock-Market-Portfolio-Optimization-
#This program attempts to optimize a user's portfolio using efficient frontier


pip install pandas_datareader
#importing python libraries
from pandas_datareader import data as web
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
#getting stock symbols/tickers in the portfolio
#FAANG- Facebook, Amazon, Apple, Netflix, Google
assets = ['FB', 'AMZN', 'AAPL', 'NFLX', 'GOOG']
#getting stock symbols/tickers in the portfolio
#FAANG- Facebook, Amazon, Apple, Netflix, Google
assets = ['FB', 'AMZN', 'AAPL', 'NFLX', 'GOOG']
#ASSIGN WEIGHTS TO PORTFOLIO: percentage of each company in the portfolio, must be equal to 1.
#Hypothetically assigning them to 0.2 so each weight is equal for each stock. 
weights= np.array ([0.2, 0.2, 0.2, 0.2, 0.2])


#GET THE STOCK STARTING DATE
#2013 because FB boomed in that specific year, entirely.
stockstartdate= '2013-01-01'
#GET STOCKS ENDING DATE (TODAY)
today= datetime.today().strftime('%Y-%m-%d')
today
#CREATE A DATAFRAME TO STORE THE ADJUSTED CLOSING PRICE OF THE STOCK
df= pd.DataFrame()

#STORE THE ADJUSTED CLOSE PRICE OF THE STOCK INTO THE df
for stock in assets:
    df[stock]= web.DataReader(stock, data_source= 'yahoo', start=stockstartdate, end=today)['Adj Close']
    

#SHOW THE DATAFRAME
df
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
#SHOW THE DAILY SIMPLE RETURNS
returns= df.pct_change()
returns
#SHOW ANNUALIZED COVARAIANCE MATRIX
cov_matrix_annual= returns.cov() * 252
cov_matrix_annual
#CALCULATE THE PORTFOLIO VARIANCE
port_variance = np.dot(weights.T, np.dot(cov_matrix_annual, weights))
port_variance
#CALCUATE THE PORTFOLIO VOLATILITY- STANDARD DEVIATION
port_volatility= np.sqrt(port_variance)
port_volatility
#ANNUAL PORTFOLIO RETURN CALCULATION
portfolioSimpleAnnualReturn = np.sum(returns.mean() * weights) * 252
portfolioSimpleAnnualReturn
#SHOW EXPECTED ANNUAL RETURN, VOLATILITY (RISK), VARIANCE

percent_var = str(round (port_variance, 2 ) * 100) + '%'
percent_vols = str (round(port_volatility, 2)* 100) + '%'
percent_ret= str(round (portfolioSimpleAnnualReturn, 2)* 100) + '%'

print ('Expected Annual return :'+ percent_ret)
print ('annual volatility or risk :'+ percent_vols)
print('Annual variance :'+ percent_var)

pip install PyPortfolioOpt

from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns

