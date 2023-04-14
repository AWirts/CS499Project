import pandas as pd
import numpy as np
from sklearn.cluster import SpectralClustering
import matplotlib.pyplot as plt

def cleanData(raw, clean):
    dataset = pd.read_csv(raw)
    dataset = dataset[['store_order_number', 'Delivery_run_completed_timestamp', 'time_delivery', 'Day_part', 'Delivery_latitude', 'Delivery_longitude']]

    drops = set()
    for label, column in dataset.items():
            index = 0
            for row in column:
                if(pd.isna(row)):
                    drops.add(index)
                index+=1

    dataset=dataset.drop(drops)
    dataset.to_csv(clean)

def plotData(dataset):
    fig = plt.figure(figsize = (10, 7))
    ax = plt.axes(projection ="3d")

    x = dataset['Delivery_latitude']
    y = dataset['Delivery_longitude']
    z = dataset['Delivery_run_completed_timestamp']

    ax.scatter(x, y, z)
    ax.set_xlabel('Latitude')
    ax.set_ylabel('Longitude')
    ax.set_zlabel('Time')
    plt.show()

def getClusters(dataset):
    ward = SpectralClustering(n_clusters=5,assign_labels='discretize',random_state=0).fit(dataset)
    label = ward.labels_
    
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


filename = 'UK_data_test.csv'
dataset = pd.read_csv(filename)
dataset['Delivery_run_completed_timestamp'] = pd.to_numeric(pd.to_datetime(dataset['Delivery_run_completed_timestamp']))

dataset = dataset[['Delivery_latitude', 'Delivery_longitude', 'Delivery_run_completed_timestamp']]
dataset = dataset.to_numpy()
dataset[:, 2] /= 10**16

label = getClusters(dataset)

dataset = pd.DataFrame(dataset, columns=['Delivery_latitude', 'Delivery_longitude', 'Delivery_run_completed_timestamp'])

dataset['Label'] = label.tolist()

plotGroupData(dataset, 'Label')

#18|20221208|0041
 