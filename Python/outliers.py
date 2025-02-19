import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.impute import SimpleImputer

dfMain = pd.read_csv('../aggregate.csv')
dfMain['Location'] = dfMain['Location'].str.strip()	

dfMainNoInvoiceLoc = dfMain[dfMain['Date Received'] != '/  /    ']

assert(len(dfMainNoInvoiceLoc) < len(dfMain))

#dfMainNoInvoiceLoc['Ring Size'] = dfMainNoInvoiceLoc['Ring Size'].replace('No Size', 0.0).astype(float)

dfDaysPivot = pd.pivot_table(dfMainNoInvoiceLoc, values=['Days', 'Location'], index='Bag #', aggfunc={'Days': np.sum, 'Location': np.count_nonzero})
dfDaysPivot.reset_index(inplace=True)

dfDaysPivot.describe()

#Elbow method to find best number of clusters
dfDaysPivot = dfDaysPivot.select_dtypes(include='number')
# Data Preprocessing
# Handling missing values
imputer = SimpleImputer(strategy='mean')
dfFilled = pd.DataFrame(imputer.fit_transform(dfDaysPivot.select_dtypes(include='number')), columns=dfDaysPivot.columns)

# Normalizing numerical features
scaler = StandardScaler()
dfScaled = pd.DataFrame(scaler.fit_transform(dfFilled.select_dtypes(include='number')), columns=dfFilled.columns)

inertia = []
cluster_options = [2, 3, 4, 5, 6]  # Replace with the range of clusters you want to try
for n_clusters in cluster_options:
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(dfScaled)
    inertia.append(kmeans.inertia_)

# Plot the inertia to see which number of clusters is best
plt.plot(cluster_options, inertia, '-o')
plt.xlabel('Number of clusters, k')
plt.ylabel('Inertia')
plt.title('Inertia of k-Means versus number of clusters')
plt.show()

# KMeans with 3 clusters
scaler = StandardScaler()
scaled_data = scaler.fit_transform(dfDaysPivot)
kmeans = KMeans(n_clusters=3, random_state=0) 
dfDaysPivot['cluster'] = kmeans.fit_predict(scaled_data)

uniqueClusters = dfDaysPivot['cluster'].unique()
colors = plt.cm.viridis(np.linspace(0, 1, len(uniqueClusters))) # Generate a colormap

plt.figure(figsize=(8, 6))

scatter = plt.scatter(dfDaysPivot['Days'], dfDaysPivot['Location'], c=dfDaysPivot['cluster'], cmap='viridis', marker='o', edgecolor='k')

#plt.scatter(x=dfDaysPivot['Days'], y=dfDaysPivot['Location'], s = 10)
plt.xlabel("Days In Production")
plt.ylabel("# Locations Visited")
plt.colorbar(label='Cluster')
plt.show()

# DBScan 
dbscan = DBSCAN(eps=10.0, min_samples=10)
dfDaysPivot['cluster'] = dbscan.fit_predict(dfDaysPivot[['Days', 'Location']])

uniqueClusters = dfDaysPivot['cluster'].unique()
colors = plt.cm.viridis(np.linspace(0, 1, len(uniqueClusters))) # Generate a colormap

plt.figure(figsize=(8, 6))

for i, clustedId in enumerate(uniqueClusters):

    cluster_data = dfDaysPivot[dfDaysPivot['cluster'] == clustedId]

    if clustedId == -1:
        plt.scatter(cluster_data['Days'], cluster_data['Location'], color='gray', label='Outliers', marker='o', edgecolor='k')
    else:
        plt.scatter(cluster_data['Days'], cluster_data['Location'], color=colors[i], label=f'Cluster {clustedId}', marker='o', edgecolor='k')


#plt.scatter(x=dfDaysPivot['Days'], y=dfDaysPivot['Location'], s = 10)
plt.xlabel("Days In Production")
plt.ylabel("# Locations Visited")
plt.legend()
plt.show()