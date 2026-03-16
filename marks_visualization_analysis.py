# 100_student(maths,science,english)_analysis

import numpy as np
import matplotlib.pyplot as plt


# viswalise the data of 50 students and their subjects(maths,science,english)
sr_no = np.arange(1,101)
numbers = sr_no.reshape((100,1))

hours = np.random.randint(1,9,size =100)

marks = np.random.randint(45,97,size = (100,3))

# shape of the marks array
print("Shape of the marks array: \n",marks.shape,"\n")


# average marks of the each subject
avg = np.mean(marks,axis = 0 )
print("Average of the each subject: \n",avg,"\n")

# average marks of the each student
avg_stu = np.mean(marks,axis = 1)
print("Average of the each student:\n",avg_stu,"\n")

sorting = np.flip(np.sort(avg_stu,axis = 0))
print(sorting,"\n")

print("Top 5 students based on their avg :",sorting[:5],"\n")

where_condition = np.where(marks>=40,"Pass","Fail")
# print("Result: \n",where_condition)


max_avg = numbers[avg_stu == np.max(avg_stu)]
print("Maximun Averge Marks: \n",max_avg,"\n")

min_avg = numbers[avg_stu == np.min(avg_stu)]
print("Minimum Averge Marks: \n",min_avg,"\n")

plt.figure(figsize = (8,4))

plt.subplot(1,3,1)
plt.title("Average marks per subject",fontsize = 10)

subjects = np.array(["Maths","Science","English"])
plt.xlabel("Name of Subjects")
plt.ylabel("Average numbers for each Subject")

plt.bar(subjects,avg,color = ["skyblue","lightgreen","pink"],alpha =0.8)

plt.subplot(1,3,2)

plt.title("Student ID vs Overall Average Marks",fontsize =10)
plt.xlabel("Sr. number of all students")
plt.ylabel("Avgerage marks of each student")

plt.plot(sr_no,avg_stu,color = "skyblue",marker = "o",mfc = "lightgreen",ms = 4,mec = "black")
plt.axhline(65)
plt.grid()


plt.subplot(1,3,3)

plt.title("Study Hours vs Overall Average Marks",fontsize = 9)
plt.xlabel("Study hours")
plt.ylabel("Average marks of all students")
plt.scatter(hours,avg_stu ,color ="skyblue",edgecolor = "black")
plt.grid()

plt.tight_layout()
plt.show()
