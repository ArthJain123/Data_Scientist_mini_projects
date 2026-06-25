import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

## Retail Project  

# ====================== Step 1 =========================

data = pd.read_excel("/content/OnlineRetail (1).xlsx")

print("First 5 Rows of Data: \n",data.head(5))

print("\nAll Columns of the data: ",data.columns)

print("\nChecking Data types stored in each column: ")
print(data.info())

# ====================== Step 2 =========================

# Converting Date (data type) column into DateTime (data type)
data["InvoiceDate"] = pd.to_datetime(data["InvoiceDate"],format = "mixed")

print("\nChecking the Nan values from dataset: \n",data.isnull().sum(),"\n")

# Deleting the Nan Values from the Data
data = data.dropna()

# Drop the data which Quantity are less than 0
data = data.drop(data[data["Quantity"] < 0].index,axis = 0)

# Drop the data which UnitPrice are less than and eqaul to 0
data = data.drop(data[data["UnitPrice"] <= 0].index,axis = 0)

# Deleting duplicates values from dataset
data = data.drop_duplicates()

# ====================== Step 3 =========================

data["Total_Price"] = data["Quantity"] * data["UnitPrice"]
print("After Making and Adding Total Price Column: \n",data)

print("\nTotal Revenue: \t",data["Total_Price"].sum())

print("\nTotal Orders: \n",data["InvoiceNo"].nunique())

print("\nTotal Customers: \n",data["CustomerID"].nunique())

# ====================== Step 4 =========================

print("\nSum of Top Selling Products: \n",(data.groupby("Description")["Quantity"].sum()).sort_values(ascending = False))

print("\nSum of Top Revenue Products: \n",data.groupby("Description")["Total_Price"].sum().sort_values(ascending = False))

print("\nThe Country which gave High Sales in Business: \n",(data.groupby("Country")["Quantity"].sum()).idxmax())

month_sal = data.groupby(data["InvoiceDate"].dt.month)["Total_Price"].sum()
print("\nMonthly Sales Analysis: \n",month_sal)

top_cus = (data.groupby("CustomerID")["Total_Price"].sum()).sort_values(ascending = False).iloc[:10]
print("\nTop 10 Customer: \n",top_cus)

# ====================== Step 5 =========================

# Manual Way to Caluted RFM for business

print("Recency: ",data["InvoiceDate"].sort_values(ascending = False).iloc[0])

print("Frequency: ",data["Quantity"].max())

print("Monetary: ",data["Total_Price"].max())

# Using Groupby Function to Calculate RFM

print("RFM Best Customer Overall: \n",data.groupby("CustomerID")[["Total_Price","Quantity","InvoiceDate"]].max().idxmax())

# ====================== Step 6 =========================

plt.style.use("dark_background")

plt.figure(figsize = (8,4))

plt.title("Monthly Sales Analysis",fontsize = 20)

plt.xlabel("Month")
plt.ylabel("Total Month Sales")

plt.plot(month_sal.index,month_sal,color = "lightgreen",label = "This Year Sales",marker = "o")

plt.grid(axis = "y")
plt.legend()
plt.show()

plt.style.use("dark_background")

plt.title("Total Revenue of 10 High Revenue Products",fontsize = 15)

x = data.groupby("Description")["Total_Price"].sum().sort_values(ascending = False).head(10)

plt.xlabel("Total Revenue")
plt.ylabel("Name of 10 High Revenue Products")

plt.barh(x.index,x,color = "pink",hatch = "/",edgecolor = "white")

plt.grid(axis = "x")
plt.show()

plt.figure(figsize = (8,4))

plt.title("Top 10 High Sales Country's",fontsize = 15)

plt.xlabel("Total Sales")
plt.ylabel("Name of All Country's")

y = data.groupby("Country")["Total_Price"].sum().sort_values(ascending = False).head(10)
print(y)

plt.barh(y.index,y,color = "skyblue",edgecolor = "white")

plt.grid(axis = "x")
plt.show()

plt.figure(figsize = (8,4))

plt.title("Top 10 Customer(Spending Rate High)",fontsize = 20)

plt.xlabel("Top 10 Customer Id")
plt.ylabel("Total Sales of Top 10 Customer")

plt.bar(top_cus.index,top_cus,width = 100,color = "orange",edgecolor = "white")

plt.grid()
plt.tight_layout()
plt.show()

# ====================== Step 7 =========================

model = LinearRegression()

x = data.loc[:,["Quantity","UnitPrice"]].head(100)

y = data["Total_Price"].head(100)

trained = model.fit(x,y)
print("Trained Model: \n",trained)

pred = model.predict(x)
print("\nPredicted Value: \n",pred)

error = (y - pred)

r2 = r2_score(y,pred)
print("\nR**2 Score of Prediced Data: \n",r2)

print(f"Score is {r2} which is closer to 1\nThis Mean the predictted Value Accuracy By Machine is Much Better or You can say Veru Good Predition")


plt.title("Predicting values by using Linear Regression ML",fontsize = 15)

plt.xlabel('100 Rows Data')
plt.ylabel("Predicted V/S Actual Total Price")

plt.plot(y.index,y,color = "pink",label = "Actual Data",marker = ".",mfc ="red")

plt.plot(y.index,pred,color = "skyblue",label = "Predicted Value",marker = ".",mfc ="red")

plt.grid(axis = "y")
plt.legend()
plt.show()
