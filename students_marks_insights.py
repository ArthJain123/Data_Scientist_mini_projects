import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

np.random.seed(2)

marks = np.random.randint(30,97,size = (500,3))

stu_class = np.random.choice([10,11,12],size = (500))

gender = np.random.choice(["Male","Female"],size = (500))

data = pd.DataFrame(dict(Math = marks[:,0],
                         Physics = marks[:,1],
                         Chemistry = marks[:,2],
                         Class = stu_class,
                         Gender = gender))

data["Total"] = np.sum(marks,axis = 1)

data["Average"] = np.mean(marks,axis = 1)

data["Grade"] = np.where(data["Average"] >= 90,"A",
                         np.where(data["Average"] >= 75,"B",
                                  np.where(data["Average"] >= 60,"C","D")))

print(data)

print("\n",np.where(data.isnull()),"\n") # Finding Nan Value

print("\n",data.info(),"\n") # Checking each coloum have correct dtype

sub_mean = np.mean(marks,axis = 0)

sub = data.columns[:3]

print("Hardest Subject: ",sub[sub_mean.argmin()])

sorted_data = data.sort_values(by = "Average",ascending = False)

print("\nTop Students: \n",sorted_data.head(6),"\n")
print("\nWeak Students: \n",sorted_data.tail(6),"\n")

# Performance by gender and class
print(data.groupby("Class")["Average"].max())
print(data.groupby("Class")["Average"].min())

print(data.groupby("Gender")["Average"].max())
print(data.groupby("Gender")["Average"].min())

# Checking Spread of Data
# all average marks points spread on x axis line and
# find mean and check the points are closer to mean or not ,
# if not then we take standard deviation less std means good but more std means bad

print("All Statical value",data["Average"].describe())


plt.style.use("dark_background")

plt.figure(figsize = (8,5))

plt.subplot(1,3,1)
plt.title("Subjects Average marks analysis",fontsize = 15)

plt.xlabel("Name of All Subjects")
plt.ylabel("Average Marks")

plt.text(sub[0],30,sub_mean[0],bbox = dict(edgecolor = "white",facecolor = "black"))
plt.text(sub[1],30,sub_mean[1],bbox = dict(edgecolor = "white",facecolor = "black"))
plt.text(sub[2],30,sub_mean[2],bbox = dict(edgecolor = "white",facecolor = "black"))

plt.bar(sub,sub_mean,color = "pink",width = 0.5,edgecolor = "white",linestyle = "-")


unique,count = np.unique(data["Grade"],return_counts = True)

plt.subplot(1,3,2)
plt.title("Grade Distribution",fontsize = 15)

plt.pie(count,labels = unique,radius = 1.5,colors = ["yellow","pink","skyblue","lightgreen"],autopct = "%0.1f%%",wedgeprops = dict(edgecolor = "white",alpha = 0.8),shadow = True)


plt.subplot(1,3,3)
plt.title("Average Marks of All Students",fontsize = 15)

plt.xlabel("Students Average Marks")
plt.ylabel("Frequency of Average Marks")

plt.hist(data["Average"],color = "pink",rwidth = 0.9,edgecolor = "white")

plt.tight_layout()
plt.savefig("project1.pdf",transparent = False,dpi = 5000)
plt.show()

plt.subplot(1,2,1)
plt.title("Subjects Average marks analysis",fontsize = 15)

plt.xlabel("Name of All Subjects")
plt.ylabel("Average Marks")

plt.text(sub[0],30,sub_mean[0],bbox = dict(edgecolor = "white",facecolor = "black"))
plt.text(sub[1],30,sub_mean[1],bbox = dict(edgecolor = "white",facecolor = "black"))
plt.text(sub[2],30,sub_mean[2],bbox = dict(edgecolor = "white",facecolor = "black"))

plt.bar(sub,sub_mean,color = "pink",width = 0.5,edgecolor = "white",linestyle = "-")

plt.subplot(1,2,2)
plt.title("Grade Distribution",fontsize = 15)

plt.pie(count,labels = unique,radius = 1.5,colors = ["yellow","pink","skyblue","lightgreen"],autopct = "%0.1f%%",wedgeprops = dict(edgecolor = "white",alpha = 0.8),shadow = True)

plt.show()

