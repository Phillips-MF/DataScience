import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense ,Dropout
from sklearn.metrics import mean_absolute_error

salary_data = pd.read_csv("Data/Salary_dataset.csv")
salary_data.drop("Unnamed: 0", axis = 1, inplace = True)
years_experience = salary_data.iloc[:,0]
salary = salary_data.iloc[:,1]

sns.histplot(data = salary_data, x = salary_data.iloc[:, 1])

sns.scatterplot(data = salary_data, x = salary, y = years_experience)

yrs_experience_training, yrs_experience_test, salary_training, salary_test = train_test_split(years_experience, salary, test_size = 0.5)

salary_predictor = Sequential() # Neural Network
#Input Layer
salary_predictor.add(Dense(units = 10, activation = 'relu', input_dim = 1))
salary_predictor.add(Dropout(0.1))
#Second layer
salary_predictor.add(Dense(units = 30, activation = 'relu'))
salary_predictor.add(Dropout(0.1))
#Third layer
salary_predictor.add(Dense(units = 15, activation = 'relu'))
salary_predictor.add(Dropout(0.1))
#Output layer
salary_predictor.add(Dense(units = 1, activation = 'relu'))
#Compilation
salary_predictor.compile(loss = 'mean_absolute_error', optimizer = 'adam', metrics = ['mean_absolute_error'])
#Training session
salary_predictor.fit(yrs_experience_training, salary_training, batch_size=10, epochs = 250)

salary_prediction = salary_predictor.predict(yrs_experience_test)

#Error
mae = mean_absolute_error(salary_test, salary_prediction)
print(round(mae))