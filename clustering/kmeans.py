import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def getClusters(dataset):
    kmeans = KMeans(n_clusters=2, random_state=0).fit(dataset)
    label = kmeans.labels_
    return label

def plotGroupData(dataset, group):
    if group == 'Delivery_run_completed_date':
        dataset.insert(2, 'Delivery_run_completed_date', pd.to_datetime(dataset['Delivery_run_completed_timestamp']).dt.date)
        dataset = dataset.groupby('Delivery_run_completed_date')
    else:
        dataset = dataset.groupby(group)

    fig = plt.figure(figsize = (10, 7))
    ax = plt.axes(projection ="3d")

    for name, group in dataset:
        x = group['Delivery_latitude']
        y = group['Delivery_longitude']
        z = group['Delivery_run_completed_timestamp']
        ax.scatter(x, y, z, label=name, s=20, edgecolor="k")

    plt.legend()
    ax.set_xlabel('Latitude')
    ax.set_ylabel('Longitude')
    ax.set_zlabel('Time')
    plt.show()

dataset = pd.read_csv('UK_data_2_clean.csv')
dataset['Delivery_run_completed_timestamp'] = pd.to_numeric(pd.to_datetime(dataset['Delivery_run_completed_timestamp']))

dataset = dataset[['Delivery_latitude', 'Delivery_longitude', 'Delivery_run_completed_timestamp']]
dataset = dataset.to_numpy()
dataset[:, 2] /= 10**16

label = getClusters(dataset)

dataset = pd.DataFrame(dataset, columns=['Delivery_latitude', 'Delivery_longitude', 'Delivery_run_completed_timestamp'])
dataset['Label'] = label.tolist()

plotGroupData(dataset, 'Label')