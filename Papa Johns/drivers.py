import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import lightgbm as ltb

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
#get data and split into 30 minutes time periods
orders = pd.DataFrame()
dataset = pd.read_csv('UK_data_2.csv')
dataset['Out_the_door_timestamp'] = pd.to_datetime(dataset['Out_the_door_timestamp'])
orders.insert(0, 'Date', dataset['Out_the_door_timestamp'].dt.date)
orders.insert(1, 'Time', dataset['Out_the_door_timestamp'].dt.floor('30Min').dt.time)
#print(orders)
cnt = orders.groupby(['Date','Time']).size().rename('Count')
orders = orders.drop_duplicates(subset=['Date','Time'])\
    .merge(cnt, left_on=['Date','Time'], right_index=True)


orders = orders.sort_values(by=['Date','Time'])
#print(orders)
#orders.to_csv('UK_orders_30Min.csv')

#convert to numbers
test = orders.apply(lambda r : pd.datetime.combine(r['Date'],r['Time']),1)
orders["date_delta"] = (pd.to_numeric((test.dt.strftime("%Y%m%d%H%M%S"))))
print(orders["date_delta"])
orders["time_delta"] = pd.to_numeric((test.dt.hour*60 + test.dt.minute))
x = np.array(orders[["date_delta", "time_delta"]])
y = np.array(orders["Count"])
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2)

# Use pip install lightgbm to install it on your system
#put into model
model = ltb.LGBMRegressor()
model.fit(xtrain, ytrain)
ypred = model.predict(xtest)
data = pd.DataFrame(data={"Predicted Orders": ypred.flatten()})
data2 = pd.DataFrame(data={"Actual Orders": ytest.flatten()})
print(data.head())
print(data2.head())

#print accuracy report
from sklearn.metrics import confusion_matrix ,classification_report 
threshold_nor=2
y_pred_nor=[0 if c<=threshold_nor else 1 for c in ypred]
print(classification_report(ytest,y_pred_nor))
print(confusion_matrix(ytest,y_pred_nor))

""" cnt = orders.groupby('Date').size().rename('Count')
result = orders.drop_duplicates(subset='Date')\
    .merge(cnt, left_on='Date', right_index=True)
print(result)
 """
