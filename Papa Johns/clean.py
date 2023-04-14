
import pandas as pd
import numpy as np
filename1 = 'UK_data_1.csv'
filename2 = 'UK_data_2.csv'
dataset = pd.read_csv(filename2)
#print(dataset)
drops = set()
for label,column in dataset.items():
        if(label=="Glympse_timestamp_live" or label=='Glympse_timestamp_arrived'):
             continue
        index = 0
        for row in column:
            if(pd.isna(row)):
                print("dropping row " + str(index))
                drops.add(index)

            index+=1

        
 
dataset=dataset.drop(drops)
dataset=dataset.assign(time_delivery=(pd.to_datetime(dataset['Delivery_run_completed_timestamp'])-
                              pd.to_datetime(dataset['Out_the_door_timestamp'])))

for row,item in dataset['time_delivery'].items():
    dataset['time_delivery'][row]=item.total_seconds()/60



dataset.to_csv('UK_data_test.csv')
print(dataset) 