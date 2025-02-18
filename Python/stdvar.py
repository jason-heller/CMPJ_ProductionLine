import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Add CSV and clean
dfMain = pd.read_csv('../aggregate.csv')
dfMain['Location'] = dfMain['Location'].str.strip()	

# Remove the AWIN location (invoice - no technical 'end date' so will mess up the data with large number of days)
dfMainNoInvoiceLoc = dfMain[dfMain['Location'] != 'AWIN']
# print(dfMain['Location'])

# Make sure we actually removed something
assert(len(dfMainNoInvoiceLoc) < len(dfMain))
# print(dfMainNoInvoiceLoc.head())

# Pivot bag id by sum of days
dfDaysPivot = pd.pivot_table(dfMainNoInvoiceLoc, values='Days', index='Bag #', aggfunc=np.sum)
dfDaysPivot = dfDaysPivot.sort_values(by='Days', ascending=False)

# print(dfDaysPivot)
daysMean = round(dfDaysPivot['Days'].mean(), 2)
daysVar = round(dfDaysPivot['Days'].var(ddof=0), 2)
daysStd = round(dfDaysPivot['Days'].std(ddof=0), 2)

print('total rows:', len(dfDaysPivot))
print('mean:', daysMean)
print('variance:', daysVar)
print('std:', daysStd)
 
# Calculating probability density function (PDF)
pdf = stats.norm.pdf(dfDaysPivot['Days'], daysMean, daysStd)

# Drawing a graph
plt.plot(dfDaysPivot['Days'], pdf)
plt.xlim([0,500])  
plt.title('Normal Distribution of Item Processing Time')
plt.xlabel("Number of Days")    
plt.ylabel("Probability Density")                
plt.grid(True, alpha=0.3, linestyle="--")
plt.show()