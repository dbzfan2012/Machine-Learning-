import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
from sklearn.linear_model import LinearRegression 
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.metrics import r2_score

x = np.array(65) 
y = np.array(70)
    
# opens file and obtains age data
file = open("age.txt")
for i in range(1552):
    age = int(file.readline())
    x = np.append(age,x)

# opens file and obtains price data
file = open("price.txt")
for i in range(1552):
    price = int(file.readline().strip())
    y = np.append(price,y)

mymodel = np.poly1d(np.polyfit(x, y, 3))

myline = np.linspace(0.1, 100, 100)

plt.scatter(x, y)
plt.plot(myline, mymodel(myline))
plt.show()

#print(plt.coef)

print("R squared value is", r2_score(y, mymodel(x)))


#Find Highest Degree R squared Score
# maxR = 0
# maxDegree = 0
# for i in range (151):
#     mymodel = np.poly1d(np.polyfit(x, y, i + 1))
#     r = r2_score(y, mymodel(x))
#     if (r > maxR):
#         maxR = r
#         maxDegree = i + 1
# print("Highest R is", str(maxR) + " at degree", str(maxDegree))






