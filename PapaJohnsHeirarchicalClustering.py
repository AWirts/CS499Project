import matplotlib.pyplot as plt
import matplotlib.colors as clrs
import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering

def cleanData(raw, clean):
    dataset = pd.read_csv(raw)
    dataset = dataset[['store_order_number', 'Out_the_door_timestamp', 'Delivery_run_completed_timestamp', 'Day_part', 'Delivery_latitude', 'Delivery_longitude']]

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
    ward = AgglomerativeClustering(n_clusters=None, distance_threshold=0.01, linkage="complete").fit(dataset)

    label = ward.labels_
    isCluster = pd.DataFrame(label).duplicated(keep=False)

    return label, isCluster

def printClusters(dataset):
    dataset = dataset.sort_values('Label').reset_index(drop=True)
    numRows = dataset.shape[0]
    i = 0

    while i in range(numRows-1):
        if dataset.at[i, 'Is_cluster']:
            for j in range(i+1, numRows):
                if dataset.at[i, 'Label'] != dataset.at[j, 'Label']:
                    print('Combined Deliveries:')
                    print(dataset.loc[i:j-1, ['store_order_number', 'Out_the_door_timestamp', 'Delivery_run_completed_timestamp', 'Delivery_latitude', 'Delivery_longitude']].to_string())
                    print('\n')
                    i = j
                    break
        else:
            i += 1

def plotClusters(dataset):
    plt.figure(1)

    ax = plt.axes(projection ="3d")

    x = dataset['Delivery_latitude']
    y = dataset['Delivery_longitude']
    z = dataset['Delivery_run_completed_float']

    c = dataset['Is_cluster']
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

def getAggregateClusters(dataset):
    dataset = dataset.sort_values('Label').reset_index(drop=True)

    numRows = dataset.shape[0]

    i = 0

    while i in range(numRows-1):
        if dataset.at[i, 'Is_cluster']:
            for j in range(i+1, numRows):
                if dataset.at[i, 'Label'] == dataset.at[j, 'Label']:
                    if dataset.at[i, 'Out_the_door_float'] > dataset.at[j, 'Out_the_door_float']:
                        dataset.at[i, 'Out_the_door_float'] = dataset.at[j, 'Out_the_door_float']
                        dataset.at[i, 'Out_the_door_timestamp'] = dataset.at[j, 'Out_the_door_timestamp']
                        dataset.at[i, 'Day_part'] = dataset.at[j, 'Day_part']
                    if dataset.at[i, 'Delivery_run_completed_float'] < dataset.at[j, 'Delivery_run_completed_float']:
                        dataset.at[i, 'Delivery_run_completed_float'] = dataset.at[j, 'Delivery_run_completed_float']
                        dataset.at[i, 'Delivery_run_completed_timestamp'] = dataset.at[j, 'Delivery_run_completed_timestamp']
                        dataset.at[i, 'Delivery_run_completed_day'] = dataset.at[j, 'Delivery_run_completed_day']
                    dataset = dataset.drop([j])
                else:
                    i = j
                    break
        else:
            i += 1

    return dataset.reset_index(drop=True)

def getNumDrivers(dataset):
    dataset['Num_drivers'] = 1

    numRows = dataset.shape[0]

    for i in range(numRows-1):
        for j in range(i+1, numRows):
            if dataset.at[i, 'Delivery_run_completed_float'] >= dataset.at[j, 'Out_the_door_float']:
                dataset.at[j, 'Num_drivers'] += 1
            else:
                break

    return dataset['Num_drivers']

def plotDeliveryDuration(dataset):
    plt.figure(2)
    ax = plt.axes()

    for index, row in dataset.iterrows():
       plt.vlines(index, row['Out_the_door_float'], row['Delivery_run_completed_float'], colors=('orange' if row['Is_cluster'] else 'darkblue'))

    plt.xlabel("Index")
    plt.ylabel("DateTime")

    days = pd.unique(dataset['Delivery_run_completed_day'])
    ax.set_yticks(pd.to_numeric(pd.to_datetime(days)))
    ax.set_yticklabels(days)

    #plt.show()

def plotNumDrivers(dataset):
    plt.figure(3)
    ax = plt.axes()

    ax.barh(dataset['Delivery_run_completed_timestamp'], dataset['Num_drivers'], height=0.01)

    plt.xlabel("Number of Drivers")
    plt.ylabel("DateTime")

raw = 'UK_data_2.csv'
clean = 'UK_data_2_clean.csv'

cleanData(raw, clean)

dataset = pd.read_csv(clean)

dataset['Out_the_door_timestamp'] = pd.to_datetime(dataset['Out_the_door_timestamp'])
dataset.insert(2, 'Out_the_door_float', pd.to_numeric(dataset['Out_the_door_timestamp']))

dataset['Delivery_run_completed_timestamp'] = pd.to_datetime(dataset['Delivery_run_completed_timestamp'])
dataset.insert(4, 'Delivery_run_completed_float', pd.to_numeric(dataset['Delivery_run_completed_timestamp']))
dataset.insert(5, 'Delivery_run_completed_day', dataset['Delivery_run_completed_timestamp'].dt.date)

points = dataset[['Delivery_latitude', 'Delivery_longitude', 'Delivery_run_completed_float']]
points = points.to_numpy()
points[:, 2] /= 10**13

label, isCluster = getClusters(points)

dataset['Label'] = label.tolist()
dataset['Is_cluster'] = isCluster.tolist()

dataset['Num_drivers'] = getNumDrivers(dataset)

plotClusters(dataset)

printClusters(dataset)
dataset = getAggregateClusters(dataset)
dataset = dataset.sort_values('Delivery_run_completed_float').reset_index(drop=True)

plotDeliveryDuration(dataset)
plotNumDrivers(dataset)

plt.show()

#dataset.to_csv('test.csv')
