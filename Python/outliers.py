import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

dfMain = pd.read_csv('../aggregate.csv')
dfMain['Location'] = dfMain['Location'].str.strip()	
dfMainNoInvoiceLoc = dfMain[dfMain['Location'] != 'AWIN']
dfMainNoInvoiceLoc['Ring Size'] = dfMainNoInvoiceLoc['Ring Size'].replace('No Size', 0.0).astype(float)

dfDaysPivot = pd.pivot_table(dfMainNoInvoiceLoc, values='Days', index='Ring Size', aggfunc='mean')
#dfDaysPivot = dfDaysPivot.sort_values(by='Days', ascending=False)

print(dfDaysPivot)
print(dfDaysPivot['Ring Size'].dtypes)

plt.scatter(x=dfDaysPivot['Days'], y=dfDaysPivot['Ring Size'], s = 100)
plt.show()