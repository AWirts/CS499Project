import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn import preprocessing

#PapaMaster = np.genfromtxt("UK_data_4.csv", delimiter=',', dtype=str)
#PapaMaster = pd.read_csv("UK_data_4.csv", index_col = 0)
PapaMaster = pd.read_csv("UK_data_4.csv")

#deleted unwarrented data
del PapaMaster[PapaMaster.columns[0]]
PapaMaster = PapaMaster.drop(columns = ["store_order_number", "To_oven_timestamp", "Initial_estimated_delivery_timestamp", "Make_time_v2", "Bake_time", "Rack_time", "Otd_time", "Glympse_timestamp_live", "Glympse_timestamp_arrived", "initial_estimated_delivery_timestamp_1", "estimated_delivery_timestamp", "Delivery_address", "Delivery_city", "Delivery_state", ])

#function to convert timestamps into useable data
def TimeToFloat(ColumnName, Array):
    for i in PapaMaster.index:
        hours = Array[ColumnName].loc[i][11] + Array[ColumnName].loc[i][12]
        minutes = Array[ColumnName].loc[i][14] + Array[ColumnName].loc[i][15]
        seconds = Array[ColumnName].loc[i][17] + Array[ColumnName].loc[i][18]
        timeMinutes = int(hours) * 60 + int(minutes) + int(seconds) / 60
        PapaMaster[ColumnName].loc[i] = timeMinutes
    Array[ColumnName] = pd.to_numeric(Array[ColumnName])

#use of functions
TimeToFloat("Business_timestamp_order_taken", PapaMaster)
TimeToFloat("Out_the_door_timestamp", PapaMaster)
TimeToFloat("Delivery_run_completed_timestamp", PapaMaster)
TimeToFloat("order_completed_timestamp", PapaMaster)
TimeToFloat("Makeline_timestamp", PapaMaster)


#deleted more uneeded data
PapaMaster = PapaMaster.drop(columns = ["Business_timestamp_order_taken", "Out_the_door_timestamp", "Make_time", "Delivery_run_completed_timestamp", "customer_quote_time", "avg_delivery_time_minutes", "Day_part", "Delivery_latitude", "Delivery_longitude"])

#print(PapaMaster)

#PapaMaster.to_csv('papatest.csv')


#Make a new column using previous data for delivery times
DeliveryTime = {"Time": []}
DeliveryTime = pd.DataFrame(DeliveryTime)

for i in PapaMaster.index:
    TimeMade = PapaMaster["Makeline_timestamp"].loc[i]
    TimeDelivered = PapaMaster["order_completed_timestamp"].loc[i]
    Time = TimeDelivered - TimeMade
    if(Time <= 0):
        Time = TimeDelivered + 60*24 - TimeMade
    DeliveryTimeTemp = {"Time": Time}
    DeliveryTime.loc[i] = DeliveryTimeTemp

#print(DeliveryTime)

#deletes column only needed for finding the total delivery time
PapaMaster = PapaMaster.drop(columns = ["order_completed_timestamp"])


#linear regression model using gradient descent
PapaLinearRegression = SGDRegressor(max_iter=1000, eta0=.05)

#scaling for use in regression
PapaMasterScalar = preprocessing.StandardScaler().fit(PapaMaster)
PapaMaster = PapaMasterScalar.transform(PapaMaster)

#linear regression training
PapaLinearRegression.fit(PapaMaster, DeliveryTime)

#prediciton
Predicted = PapaLinearRegression.predict(PapaMaster)


#absolute error calculation
TotalWrong = 0
TotalNumTimes = 0

for i in DeliveryTime.index:
    TotalWrong = TotalWrong + abs(DeliveryTime.loc[i] - Predicted[i])
    TotalNumTimes = TotalNumTimes + 1

ABSError = TotalWrong / TotalNumTimes

print(ABSError)

DeliveryTime['Predicted'] = Predicted

print(DeliveryTime)

#DeliveryTime.to_csv('papatest.csv')