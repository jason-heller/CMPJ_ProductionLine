import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.cluster import DBSCAN

dfMain = pd.read_csv('../aggregate.csv')
dfMain['Location'] = dfMain['Location'].str.strip()	

dfMainNoInvoiceLoc = dfMain[dfMain['Date Received'] != '/  /    ']

assert(len(dfMainNoInvoiceLoc) < len(dfMain))

#dfMainNoInvoiceLoc['Ring Size'] = dfMainNoInvoiceLoc['Ring Size'].replace('No Size', 0.0).astype(float)

dfDaysPivot = pd.pivot_table(dfMainNoInvoiceLoc, values=['Days', 'Location'], index='Bag #', aggfunc={'Days': np.sum, 'Location': np.count_nonzero})
dfDaysPivot.reset_index(inplace=True)

dfDaysPivot.describe()

dbscan = DBSCAN(eps=11.0, min_samples=10) # Adjust eps and min_samples as needed
dfDaysPivot['cluster'] = dbscan.fit_predict(dfDaysPivot[['Days', 'Location']])

uniqueClusters = dfDaysPivot['cluster'].unique()
colors = plt.cm.viridis(np.linspace(0, 1, len(uniqueClusters))) # Generate a colormap

plt.figure(figsize=(8, 6))

for i, clustedId in enumerate(uniqueClusters):

    cluster_data = dfDaysPivot[dfDaysPivot['cluster'] == clustedId]

    if clustedId == -1:
        plt.scatter(cluster_data['Days'], cluster_data['Location'], color='gray', label='Outliers')
    else:
        plt.scatter(cluster_data['Days'], cluster_data['Location'], color=colors[i], label=f'Cluster {clustedId}')


#plt.scatter(x=dfDaysPivot['Days'], y=dfDaysPivot['Location'], s = 10)
plt.xlabel("Days In Production")
plt.ylabel("# Locations Visited")
plt.legend()
plt.show()