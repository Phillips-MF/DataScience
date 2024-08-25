import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense

data = pd.read_csv('student_performance.csv')
data.drop('StudentID', axis = 'columns', inplace = True)
data.drop('Gender', axis = 1, inplace = True)
data.drop('Name', axis = 1, inplace = True)

#Label Encoder for ParentalSupport column
le = LabelEncoder()
data['ParentalSupport']
data['ParentalSupport']= le.fit_transform(data['ParentalSupport'])

#Predictioners / Class
predictioners = data.iloc[:, 0:5]
classes = data['FinalGrade']

#Training / Test
predictioners_training, predictioners_test, classes_training, classes_test = train_test_split(predictioners, classes, test_size=0.4)

#Neural Network
predictioners_training.shape

#Input Layer
performance_predictor = Sequential()
performance_predictor.add(Dense(units = 3, activation = 'relu', input_dim = 5))

#First Hidden Layer
performance_predictor.add(Dense(units = 3, activation = 'relu'))

#Output Layer
performance_predictor.add(Dense(units = 1, activation = 'linear'))

#Compile
performance_predictor.compile(loss = 'mean_absolute_error', optimizer = 'adam', metrics = ['mean_absolute_error'])

#Training session
performance_predictor.fit(predictioners_training, classes_training, batch_size=10, epochs = 100)

#Test
prediction = performance_predictor.predict(predictioners_test)

classes_test.mean()
prediction.mean()
