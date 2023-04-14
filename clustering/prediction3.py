import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.linear_model import LinearRegression

#Load data
data = pd.read_csv('UK_data_2_clean.csv')

#create dataframe with only relevant columns
df = data[['Delivery_latitude', 'Delivery_longitude', 'Delivery_run_completed_timestamp', 'avg_delivery_time_minutes']]

#convert timestamp to datetime
df['Delivery_run_completed_timestamp'] = pd.to_datetime(df['Delivery_run_completed_timestamp'])

#cluster locations using heirarchical clustering
X = df[['Delivery_latitude', 'Delivery_longitude']].values
cluster = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
cluster.fit_predict(X)
df['cluster'] = cluster.labels_

#calculate avg delivery time for each cluster
cluster_avg_delivery_time = df.groupby('cluster')['avg_delivery_time_minutes'].mean().values

# Hierarchical clustering plot
plt.figure(figsize=(10, 7))
plt.scatter(X[:,0], X[:,1], c=cluster.labels_, cmap='rainbow')
plt.title('Delivery Hotspots')
plt.xlabel('Delivery Latitude')
plt.ylabel('Delivery Longitude')
plt.show()

#use linear regression to cluster avg delivery time to find optimal number of drivers
X = np.arange(len(cluster_avg_delivery_time)).reshape(-1, 1)
y = cluster_avg_delivery_time.reshape(-1, 1)
regressor = LinearRegression()
regressor.fit(X, y)
optimal_drivers = int(np.ceil(regressor.predict([[len(cluster_avg_delivery_time)]])[0][0]))

# Linear regression plot
plt.figure(figsize=(10, 7))
plt.scatter(np.arange(len(cluster_avg_delivery_time)), cluster_avg_delivery_time, color='blue')
plt.plot(np.arange(len(cluster_avg_delivery_time)), regressor.predict(X), color='red')
plt.title('Average Delivery Time by Cluster')
plt.xlabel('Cluster')
plt.ylabel('Average Delivery Time (minutes)')
plt.show()

print(f'Optimal number of delivery drivers needed: {optimal_drivers}')

