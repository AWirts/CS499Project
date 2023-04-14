import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

data = pd.read_csv("UK_data_2_clean.csv")
data["Delivery_run_completed_timestamp"] = pd.to_datetime(data["Delivery_run_completed_timestamp"])
data["hour"] = data["Delivery_run_completed_timestamp"].dt.hour
data["minute"] = data["Delivery_run_completed_timestamp"].dt.minute
features = ["Delivery_latitude", "Delivery_longitude", "hour", "minute"]
target = "avg_delivery_time_minutes"
X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.2, random_state=42)
gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gbr.fit(X_train, y_train)
y_pred = gbr.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error:", mae)
threshold = 5
drivers_required = int(np.ceil(data[target].mean() / threshold))
print("Optimal number of delivery drivers required:", drivers_required)


#scatter plot of delivery locations
plt.scatter(data["Delivery_longitude"], data["Delivery_latitude"])
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Delivery Locations")
plt.show()



#scatterplot of predicteed vs actual delivery times
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Delivery Time (minutes)")
plt.ylabel("Predicted Delivery Time (minutes)")
plt.title("Predicted vs Actual Delivery Times")
plt.show()

#bar chart for number of drivers
plt.bar("Drivers Required", drivers_required)
plt.xlabel("Number of Drivers")
plt.ylabel("Threshold (minutes)")
plt.title("Optimal Number of Delivery Drivers")
plt.show()