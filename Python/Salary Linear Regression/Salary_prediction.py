import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense

salary_data = pd.read_csv("Data/Salary_dataset.csv")
salary_data.drop("Unnamed: 0", axis = 1, inplace = True)
years_experience = salary_data.iloc[:,0]

sns.histplot(data = salary_data, x = salary_data.iloc[:, 1])

data_salary_normal = np.random.normal(salary_data.iloc[:,1])

sns.histplot(x=data_salary_normal)
sns.scatterplot(x = years_experience, y = data_salary_normal)


len(salary_data.iloc[:,1])
len(data_salary_normal)

predictioner = years_experience
real_salary = data_salary_normal

predictioner_training, predictioner_test, real_salary_training, real_salary_test = train_test_split(predictioner, real_salary, test_size = 0.5)

salary_predictor = Sequential() # Neural Network

#Input Layer
salary_predictor.add(Dense(units = 7, activation = 'relu', input_dim = 1))

#Second layer
salary_predictor.add(Dense(units = 7, activation = 'relu'))

#Output layer
salary_predictor.add(Dense(units = 1, activation = 'relu'))

#Compilation
salary_predictor.compile(loss = 'mean_absolute_error', optimizer = 'adam', metrics = ['mean_absolute_error'])

#Training session
salary_predictor.fit(predictioner_training, real_salary_training, batch_size=10, epochs = 200)

salary_prediction = salary_predictor.predict(predictioner_test)