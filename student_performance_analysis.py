import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

np.random.seed(5)

marks = np.random.randint(35,101,size = (100,5))
# print(marks)

stu_id = np.arange(1,101)

data = pd.DataFrame(dict(ID = stu_id,
                         Math = marks[:,0],
                         Physics = marks[:,1],
                         Chemistry = marks[:,2],
                         IT = marks[:,3],
                         English = marks[:,4]))

total = np.sum(marks,axis = 1)

stu_avg = np.mean(marks,axis = 1)

grade = np.where(stu_avg >= 90,"A",
                 np.where(stu_avg >= 75,"B",
                          np.where(stu_avg >=60,"C"
                          ,np.where(stu_avg >= 50,"D","E"))))
# Add New column of Total
data["Total"] = total

# Add New column of Average
data["Average"] = stu_avg

# Add New column of Grade
data["Grade"] = grade

# Add New column of Result
data["Result"] = np.where(stu_avg >50 , "Pass","Fail")
print(data)

sorted_data = data.sort_values(by = "Total",ascending = False)
# print(sorted_data)

top = sorted_data.head(5)
print("Topper: ",top)

# Each Subject Average
print("Subjects Average: \n",np.mean(marks,axis = 0),"\n")

unique,counts = np.unique(data["Grade"],return_counts = True)
print(f"{unique}\n{counts},\n")

# Math topper
math_max =  data.loc[data["Math"].idxmax()]
print("Highest Score in Maths: \n",math_max,"\n")

# Physics Topper
physics_max = data.loc[data["Physics"].idxmax()]
print("Highest Score in Physics: \n",physics_max,"\n")

# Chemistry Topper
chemistry_max = data.loc[data["Chemistry"].idxmax()]
print("Highest Score in Chemistry: \n",chemistry_max,"\n")

# It Topper
it_max = data.loc[data["IT"].idxmax()]
print("Highest Score in It: \n",it_max,"\n")

# English Topper
english_max = data.loc[data["English"].idxmax()]
print("Highest Score in English: \n",english_max,"\n")

fail_stu = data[data["Average"]<50]
print("Fail Student: \n",fail_stu,"\n\n")


plt.figure(figsize = (8,5))

plt.subplot(2,2,1)

plt.title("Subject Average Chart",fontsize = 20)

sub = ["Maths","Physics","Chemistry","IT","English"]
plt.xlabel("Name of all subjects")

sub_avg = np.mean(marks,axis = 0)
plt.ylabel("Average marks of all subjects")

plt.bar(sub,sub_avg,color = "skyblue",edgecolor = "black",label = "Students Performance",width = 0.8)
plt.legend()


plt.subplot(2,2,2)

plt.title("Grade Distribution on Bar Graph",fontsize = 15)

plt.xlabel("All Grades")
plt.ylabel("Frequency of Grades")

plt.bar(unique,counts,color = "pink",edgecolor = "black",width = 0.8,label = "Students Performance")

plt.legend()


plt.subplot(2,2,3)

plt.title("Grade Distibution on Pie Chart",fontsize = 15)

plt.pie(counts,labels = unique,colors = ["pink","skyblue","lightgreen","yellow"],autopct = "%0.1f%%",wedgeprops = dict(edgecolor = "black"),radius = 1.4)


plt.subplot(2,2,4)

plt.title("Average Marks of all Students",fontsize = 15)

x = np.arange(50,81,5)

plt.xlabel("All Students")
plt.ylabel("Frequency of their Marks")

plt.hist(stu_avg,bins = x,color = "lightgreen",rwidth = 0.9,edgecolor = "black",histtype = "bar")


plt.tight_layout()
plt.show()
