from sklearn import linear_model
import pandas as pd
import statsmodels.api as sm

data = pd.read_csv('manydatas.csv')

x = data[['Bathrooms', 'Bedrooms', 'Fireplaces Total', 'Lot Square Footage', 'Parking Covered Total', 
          'Square Footage', 'Square Footage Unfinished', 'Age', 'Original Price', 'City']]
y = data['Selling Price']

line = linear_model.LinearRegression().fit(x,y)

#Prints out the line's equation
equation = ""
for i in range (len(line.coef_)):
    equation = equation + str(line.coef_[i]) + "x" + str(i + 1) + " + "
print("The equation is:" + "\n" + "y = " + equation[0:len(equation) - 3] + " + 69161.26422337")

#Prints out the r squared score
X1 = sm.add_constant(x)
result = sm.OLS(y, X1).fit()
print("The R squared value is: " + str(result.rsquared))

#Predicts based on input
a = float(input("Bathrooms: "))
b = float(input("Bedrooms: "))
c = float(input("Fireplaces: "))
d = float(input("Lot Size: "))
e = float(input("Parking Covered: "))
f = float(input("Square Footage: "))
g = float(input("Square Footage Unfinished: "))
h = float(input("Age: "))
i = float(input("Original Price: "))
j = float(input("City: "))
print("Prediction is... " + str(line.predict([[a,b,c,d,e,f,g,h,i,j]])))

# z = int(input("give repeat num: "))
# print((line.predict([[a,b,c,d,e,f,g,h,i,j]])) - ((z * line.coef_[0]) + (z * line.coef_[1]) + (z * line.coef_[2]) + (z * line.coef_[3]) 
#+ (z * line.coef_[4]) + (z * line.coef_[5]) + (z * line.coef_[6]) + (z * line.coef_[7]) + (z * line.coef_[8]) + (z * line.coef_[9])))



#[69161.26422337] is the intercept



