import matplotlib.pyplot as plt
import matplotlib.colors as clrs
import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering

def cleanData(raw, clean):
    dataset = pd.read_csv(raw)
    dataset = dataset[['store_order_number', 'Out_the_door_timestamp', 'Delivery_run_completed_timestamp', 'avg_delivery_time_minutes', 'Day_part', 'Delivery_latitude', 'Delivery_longitude']]

    drops = set()
    for label, column in dataset.items():
            index = 0
            for row in column:
                if(pd.isna(row)):
                    drops.add(index)
                index+=1

    dataset=dataset.drop(drops)
    dataset.to_csv(clean)

def getClusters(dataset):
    ward = AgglomerativeClustering(n_clusters=None, distance_threshold=0.001, linkage="complete").fit(dataset)

    label = ward.labels_
    isCluster = pd.DataFrame(label).duplicated(keep=False)

    return label, isCluster

def plotClusters(dataset):
    plt.figure(1, figsize = (10, 7))
    ax = plt.axes(projection ="3d")

    x = dataset['Delivery_latitude']
    y = dataset['Delivery_longitude']
    z = dataset['Delivery_run_completed_float']

    c = dataset['IsCluster']
    cmap = clrs.ListedColormap(['darkblue', 'orange'])

    scatter = ax.scatter(x, y, z, c=c, cmap=cmap, s=20, edgecolor="k")
    ax.set_xlabel('Latitude')
    ax.set_ylabel('Longitude')
    ax.set_zlabel('DateTime')

    days = pd.unique(dataset['Delivery_run_completed_day'])
    ax.zaxis.set_ticks(pd.to_numeric(pd.to_datetime(days)))
    ax.zaxis.set_ticklabels(days)

    plt.legend(*scatter.legend_elements(prop='colors'), title="Is Cluster")

    #plt.show()

def plotDeliveryDuration(dataset):
    plt.figure(2)
    ax = plt.axes()

    for index, row in dataset.iterrows():
       plt.vlines(index, row['Out_the_door_float'], row['Delivery_run_completed_float'], colors=('orange' if row['IsCluster'] else 'darkblue'), label=('Yes' if row['IsCluster'] else 'No'))

    plt.xlabel("Index")
    plt.ylabel("DateTime")

    days = pd.unique(dataset['Delivery_run_completed_day'])
    ax.set_yticks(pd.to_numeric(pd.to_datetime(days)))
    ax.set_yticklabels(days)

    plt.legend()

    #plt.show()

raw = 'UK_data_2.csv'
clean = 'UK_data_2_clean.csv'

#cleanData(raw, clean)

dataset = pd.read_csv(clean)

dataset['Out_the_door_timestamp'] = pd.to_datetime(dataset['Out_the_door_timestamp'])
dataset.insert(2, 'Out_the_door_float', pd.to_numeric(dataset['Out_the_door_timestamp']))

dataset['Delivery_run_completed_timestamp'] = pd.to_datetime(dataset['Delivery_run_completed_timestamp'])
dataset.insert(4, 'Delivery_run_completed_float', pd.to_numeric(dataset['Delivery_run_completed_timestamp']))
dataset.insert(5, 'Delivery_run_completed_day', dataset['Delivery_run_completed_timestamp'].dt.date)

points = dataset[['Delivery_latitude', 'Delivery_longitude', 'Delivery_run_completed_float']]
points = points.to_numpy()
points[:, 2] /= 10**16

label, isCluster = getClusters(points)

dataset['Label'] = label.tolist()
dataset['IsCluster'] = isCluster.tolist()

plotClusters(dataset)
plotDeliveryDuration(dataset)

plt.show()