import matplotlib.pyplot as plt
import matplotlib.colors as clrs
from matplotlib.lines import Line2D
import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering

#writes csv for cleaned data
def cleanData(raw, clean):

    #reads raw order data from 'raw' csv file
    dataset = pd.read_csv(raw)

    #retrieves relevant columns
    dataset = dataset[['store_order_number', 'Out_the_door_timestamp', 'Delivery_run_completed_timestamp', 'Day_part', 'Delivery_latitude', 'Delivery_longitude']]

    #drops rows with null values
    drops = set()
    for label, column in dataset.items():
            index = 0
            for row in column:
                if(pd.isna(row)):
                    drops.add(index)
                index+=1

    dataset=dataset.drop(drops)

    #writes to 'clean' csv file
    dataset.to_csv(clean)

#gets clustered deliveries based on given columns
def getClusters(dataset):

    #clusters deliveries within 'distance_threshold' of eachother
    #linkage ‘complete’ uses the maximum distances between all observations of two sets to determine clustering
    ward = AgglomerativeClustering(n_clusters=None, distance_threshold=0.5, linkage="complete").fit(dataset)

    #lists number designation of cluster assignment
    label = ward.labels_

    #lists boolean value indicating if delivery is clustered
    isCluster = pd.DataFrame(label).duplicated(keep=False)

    return label, isCluster

#prints clustered deliveries
def printClusters(dataset):

    #sorts deliveries by cluster designation so that deliveries of the same cluster are made adjacent
    dataset = dataset.sort_values('Label').reset_index(drop=True)

    numRows = dataset.shape[0]
    i = 0

    #for each delivery in data set except the last
    while i in range(numRows-1):

        #if the delivery is clustered
        if dataset.at[i, 'Is_cluster']:

            #for each subsequent delivery
            for j in range(i+1, numRows):

                #until delivery is not in current cluster
                if dataset.at[i, 'Label'] != dataset.at[j, 'Label']:

                    #print data for deliveries in current cluster
                    print('Clustered Deliveries:')
                    print(dataset.loc[i:j-1, ['store_order_number', 'Out_the_door_timestamp', 'Delivery_run_completed_timestamp', 'Delivery_latitude', 'Delivery_longitude']].to_string())
                    print('\n')

                    #continue from next delivery not in current cluster
                    i = j
                    break
        else:
            i += 1

#gets deliveries as if clustered deliveries are combined into one
def getAggregateClusters(dataset):
    dataset['Num_in_cluster'] = 1

    #sorts deliveries by cluster designation so that deliveries of the same cluster are made adjacent
    dataset = dataset.sort_values('Label').reset_index(drop=True)

    numRows = dataset.shape[0]
    i = 0

    #for each delivery in data set except the last
    while i in range(numRows-1):

        #if the delivery is clustered
        if dataset.at[i, 'Is_cluster']:

            #for each subsequent delivery
            for j in range(i+1, numRows):

                #if delivery is in current cluster
                if dataset.at[i, 'Label'] == dataset.at[j, 'Label']:

                    #assigns 'out the door' time as earliest from among clustered deliveries
                    if dataset.at[i, 'Out_the_door_float'] > dataset.at[j, 'Out_the_door_float']:
                        dataset.at[i, 'Out_the_door_float'] = dataset.at[j, 'Out_the_door_float']
                        dataset.at[i, 'Out_the_door_timestamp'] = dataset.at[j, 'Out_the_door_timestamp']
                        dataset.at[i, 'Day_part'] = dataset.at[j, 'Day_part']

                    #assigns 'delivery complete' time as latest from among clustered deliveries
                    if dataset.at[i, 'Delivery_run_completed_float'] < dataset.at[j, 'Delivery_run_completed_float']:
                        dataset.at[i, 'Delivery_run_completed_float'] = dataset.at[j, 'Delivery_run_completed_float']
                        dataset.at[i, 'Delivery_run_completed_timestamp'] = dataset.at[j, 'Delivery_run_completed_timestamp']
                        dataset.at[i, 'Delivery_run_completed_day'] = dataset.at[j, 'Delivery_run_completed_day']

                    #drops deliveries that were combined into cluster
                    dataset = dataset.drop([j])
                else:
                    dataset.at[i, 'Num_in_cluster'] = j-i
                    i = j
                    break
        else:
            i += 1

    return dataset.reset_index(drop=True)

#displays 3D scatter plot for deliveries based on latitude, longitude, and delivery completed time
def plotClusters(dataset):

    #plot title
    plt.figure('Delivery Clusters')

    #makes 3D
    ax = plt.axes(projection ="3d")

    #delivery data used for coordinates
    x = dataset['Delivery_latitude']
    y = dataset['Delivery_longitude']
    z = dataset['Delivery_run_completed_float']

    #color indicates if delivery is clustered 
    c = dataset['Is_cluster']
    cmap = clrs.ListedColormap(['darkblue', 'orange'])

    #creates scatter points
    scatter = ax.scatter(x, y, z, c=c, cmap=cmap, s=20, edgecolor="k")

    #labels xyz axes
    ax.set_xlabel('Latitude')
    ax.set_ylabel('Longitude')
    ax.set_zlabel('DateTime')

    #repopulates 'DateTime' axis to show days instead of float value used as z coordinate
    days = pd.unique(dataset['Delivery_run_completed_day'])
    ax.zaxis.set_ticks(pd.to_numeric(pd.to_datetime(days)))
    ax.zaxis.set_ticklabels(days)

    #plot legend which indicates color of clustered deliveries
    plt.legend(*scatter.legend_elements(prop='colors'), title="Is Cluster")

#displays line graph for delivery durations
def plotDeliveryDuration(dataset):

    #graph title
    plt.figure('Delivery Durations')

    ax = plt.axes()

    #for each delivery plot vertical line with start and end points at 'out the door' and 'delivery complete' times respectively
    for index, row in dataset.iterrows():
       plt.vlines(index, row['Out_the_door_float'], row['Delivery_run_completed_float'], colors=('orange' if row['Is_cluster'] else 'darkblue'))

    #label axes
    plt.xlabel("Index")
    plt.ylabel("DateTime")

    #repopulates 'DateTime' axis to show days instead of float value used as z coordinate
    days = pd.unique(dataset['Delivery_run_completed_day'])
    ax.set_yticks(pd.to_numeric(pd.to_datetime(days)))
    ax.set_yticklabels(days)

    #plot legend which indicates color of clustered deliveries
    legend_elements = [Line2D([0], [0], color='darkblue', label='0'),
                   Line2D([0], [0], color='orange', label='1')]
    ax.legend(handles=legend_elements, title="Is Cluster")

#gets number of drivers needed at the beginning of each delivery
def getNumDrivers(dataset):

    #each delivery requires minimum of 1 driver
    dataset['Num_drivers'] = 1

    numRows = dataset.shape[0]

    #for each delivery except last
    for i in range(numRows-1):

        #for each subsequent delivery
        for j in range(i+1, numRows):

            #if a new delivery begins while current delivery is ongoing, 'num driver' incremented for new delivery
            if dataset.at[i, 'Delivery_run_completed_float'] >= dataset.at[j, 'Out_the_door_float']:
                dataset.at[j, 'Num_drivers'] += 1
            else:
                break

    return dataset['Num_drivers']

#displays bargraph for number of delivery drivers needed at the beginning of each delivery
def plotNumDrivers(dataset):

    #graph title
    plt.figure('Optimal Number of Drivers')

    ax = plt.axes()

    #graphs horizontal bars where their height is the number of drivers required
    ax.barh(dataset['Delivery_run_completed_timestamp'], dataset['Num_drivers'], height=0.01)

    #labels axes
    plt.xlabel("Number of Drivers")
    plt.ylabel("DateTime")

#prints a summary of model conclusions
def printSummary(dataset):
    numOrders = 0
    numRows = dataset.shape[0]

    #calculates how many deliveries there were originaly
    for i in range(numRows):
        numOrders += dataset.at[i, 'Num_in_cluster']

    #prints the reduction of deliveries from clustering
    print('Clustering reduced', numOrders, 'deliveries to', numRows, 'deliveries.\n')

    #prins count of cluster sizes
    for index, values in dataset['Num_in_cluster'].value_counts().iteritems():
        print(values, 'deliveries had', index, 'orders.')

    print('\n')

    #prints max and average drivers needed during each part of the day
    for name, group in dataset.groupby('Day_part'):
        #print(name, group['Num_rivers'].max())
        print(name, 'had max drivers of', group['Num_drivers'].max(), 'and average of', group['Num_drivers'].mean())

#data files names
raw = 'UK_data_2.csv'
clean = 'UK_data_2_clean.csv'

#writes file for clean data
cleanData(raw, clean)

#retrieves clean data
dataset = pd.read_csv(clean)

#converts datetime strings to datetime objects and float numbers
dataset['Out_the_door_timestamp'] = pd.to_datetime(dataset['Out_the_door_timestamp'])
dataset.insert(2, 'Out_the_door_float', pd.to_numeric(dataset['Out_the_door_timestamp']))
dataset['Delivery_run_completed_timestamp'] = pd.to_datetime(dataset['Delivery_run_completed_timestamp'])
dataset.insert(4, 'Delivery_run_completed_float', pd.to_numeric(dataset['Delivery_run_completed_timestamp']))

#gets date of 'delivery completion' for use in graphs and plots
dataset.insert(5, 'Delivery_run_completed_day', dataset['Delivery_run_completed_timestamp'].dt.date)

#gets columns used in clustering
points = dataset[['Delivery_latitude', 'Delivery_longitude', 'Delivery_run_completed_float', 'Out_the_door_float']]
points = points.to_numpy()

#normalizes float values
points[:, 2] /= 10**12
points[:, 3] /= 10**12

#gets cluster designations of delivery and boolean for if it is clustered
label, isCluster = getClusters(points)
dataset['Label'] = label.tolist()
dataset['Is_cluster'] = isCluster.tolist()

#prints clusters
printClusters(dataset)

#combines clustered deliveries
dataset = getAggregateClusters(dataset)

#sorts data in order of 'out the door' time
dataset = dataset.sort_values('Out_the_door_float').reset_index(drop=True)

#plots deliveries and their durations
plotClusters(dataset)
plotDeliveryDuration(dataset)

#gets the number of drivers required and graphs it
dataset['Num_drivers'] = getNumDrivers(dataset)
plotNumDrivers(dataset)

#prints a summary of conclusions
printSummary(dataset)

#shows all plots and graphs
plt.show()