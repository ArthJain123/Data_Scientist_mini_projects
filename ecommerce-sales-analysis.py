import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression


# ============================== Step 1 =================================

data = pd.read_csv("/content/superstore_dataset.csv")

print(data.shape)

# ============================== Step 2 =================================

print("\nAll Columns Names: \n",data.columns)

print("\nTop 10 Rows of Data: \n",data.head(10))

print("\nDtypes of All Columns: ")
print(data.info())

print("\nChecking for Nan Values: \n",(data.isnull()).sum())

# ============================== Step 3 =================================

data["order_date"] = pd.to_datetime(data["order_date"],format = 'mixed')
print("\nAfter Changing the order_date to datetime: \n",data["order_date"].head(5))

# Data with no Duplicate value
data = data.drop_duplicates()
print(data.shape)

# Adding New Column of Revenue
data["Revenue"] =  data["sales"]
print(data.loc[:,["sales","Revenue"]])

# ============================== Step 4 =================================

print("\nTotal Revenue: \n",data["Revenue"].sum())

print("\nTotal Profit: \n",data["profit"].sum())

print("\nHighest Selling Category: \n",data["category"].value_counts().idxmax())

print("\nHighest Profitable Product: \n",(data.groupby("product_name")["profit"].sum()).idxmax(),"\n\n\n\n")

# ============================== Step 5 =================================

plt.style.use("dark_background")

plt.figure(figsize  = (12,8))

plt.subplot(2,2,1)

plt.title("Total Revenue over the Time",fontsize = 12)

arr = data.sort_values(by = "order_date")
plt.xlabel("Dates")

plt.ylabel("Total Revenue")

plt.plot(arr["order_date"],arr["Revenue"],color = "skyblue",marker = "o",mec = "black",mew = 0.5,label = "Total Revenue: " + str(data["Revenue"].sum()))

plt.legend(loc = "lower left",shadow = True)
plt.grid()


plt.subplot(2,2,2)

plt.title("Analysis Between Category And Revenue",fontsize = 12)

x = data.groupby("category")["Revenue"].sum()
print(x)

for val in range(3):
  plt.text(x.index[val],15,x.iloc[val],bbox = dict(facecolor = "black",edgecolor = "white"))


plt.xlabel("All Category Type")
plt.ylabel("Total Revenue")


plt.bar(x.index,x,color = "pink")


plt.subplot(2,2,3)

plt.title("Analysis Between Sales and Quantity",fontsize = 12)

plt.xlabel("Per Quantity")
plt.ylabel("Sales Generate Profit")

plt.scatter(data["quantity"],data["sales"],color = "violet",edgecolor = "black",linewidth = 0.7,label = "Total Revenue: " + str(data["Revenue"].sum()))

plt.legend(shadow = True)
plt.grid()


plt.subplot(2,2,4)

plt.title("Sales Trend Over The Time",fontsize = 12)

plt.xlabel("Total Profit Over The Time")
plt.ylabel("Total Sales")

plt.plot(arr["order_date"],arr["sales"],color = "lightgreen",marker = "o",mec = "black",label = "Average Sales" + str(data["sales"].mean()))

plt.legend(loc = "lower left",shadow = True)
plt.grid()

plt.tight_layout()
plt.show()



model = LinearRegression()

X = data.loc[:,["discount","quantity"]]
y = data["sales"]

model.fit(X,y)

print("\n\n\n\n",model.predict([[5,2]]))

print(data.loc[:,["discount","quantity","sales"]].head(1))
