import pandas
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as math
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
''' 
The following is the starting code for path1 for data reading to make your first step easier.
'dataset_1' is the clean data for path1.
'''
dataset_1 = pandas.read_csv('NYC_Bicycle_Counts_2016_Corrected.csv')
shuffled = dataset_1.to_numpy()
dataset_1['Brooklyn Bridge']      = pandas.to_numeric(dataset_1['Brooklyn Bridge'].replace(',','', regex=True))
dataset_1['Manhattan Bridge']     = pandas.to_numeric(dataset_1['Manhattan Bridge'].replace(',','', regex=True))
dataset_1['Queensboro Bridge']    = pandas.to_numeric(dataset_1['Queensboro Bridge'].replace(',','', regex=True))
dataset_1['Williamsburg Bridge']  = pandas.to_numeric(dataset_1['Williamsburg Bridge'].replace(',','', regex=True))
dataset_1['High Temp']  = pandas.to_numeric(dataset_1['High Temp'].replace(',','', regex=True))
dataset_1['Low Temp']  = pandas.to_numeric(dataset_1['Low Temp'].replace(',','', regex=True))
dataset_1['Precipitation']  = pandas.to_numeric(dataset_1['Precipitation'].replace(',','', regex=True))
#print(shuffled.to_string()) #This line will print out your data

# Break Down and Fix
##Convert to numpy arrays
brooklyn = dataset_1['Brooklyn Bridge']
brooklyn = brooklyn.to_numpy()

manhattan = dataset_1['Manhattan Bridge']
manhattan = manhattan.to_numpy()

qeensboro = dataset_1['Queensboro Bridge']
qeensboro = qeensboro.to_numpy()

williamsburg = dataset_1['Williamsburg Bridge']
williamsburg = williamsburg.to_numpy()

high_temp = dataset_1['High Temp']
high_temp = high_temp.to_numpy()

low_temp = dataset_1['Low Temp']
low_temp = low_temp.to_numpy()

precipitation = dataset_1['Precipitation']
precipitation = precipitation.to_numpy()

day = dataset_1['Day']
day = day.to_numpy()

date = dataset_1['Date']
date = date.to_numpy()

array_data = [brooklyn, manhattan, qeensboro, williamsburg]
array_atr = [low_temp, high_temp, precipitation, day]
avg_temp = np.divide((low_temp + high_temp), 2)

# We are using 80% data for training.
degrees = [1, 2, 3, 4, 5, 6, 7]
a_list = np.array(list(range(1, round(len(brooklyn) * 0.8) + 1))) #First 80% of data
a_lis = np.array(list(range(round(len(brooklyn) * 0.8), len(brooklyn)))) #last 20% of data
predict = [0] * len(degrees)
models = [0] * len(array_data)
degree_out = [0] * len(array_data)
for i in range(len(array_data)):
    holder = 1000000000000
    train = array_data[i][:(int((len(array_data[i])*0.8)))]
    test = array_data[i][(int((len(array_data[i])*0.8))):]
    for j in range(len(degrees)):
        poly = PolynomialFeatures(degree=degrees[j], include_bias=False)
        poly_features = poly.fit_transform(a_list.reshape(-1, 1))
        poly_reg_model = LinearRegression()
        poly_reg_model.fit(poly_features, train)
        test_p = PolynomialFeatures(degree=degrees[j], include_bias=False)
        test_features = test_p.fit_transform(a_lis.reshape(-1, 1))
        predict[j] = poly_reg_model.predict(test_features)
        mse_test = math.mean_squared_error(test, predict[j])
        if mse_test < holder:
            holder = mse_test
            models[i] = poly_reg_model
            degree_out[i] = degrees[j]
    print(holder)
    titl = ['Brooklyn', 'Manhattan', 'Qeensboro', 'Williamsburg']
for i in range(len(models)):
    range_x = np.array(list(range(len(brooklyn))))
    set = PolynomialFeatures(degree=degree_out[i], include_bias=False)
    features = set.fit_transform(range_x.reshape(-1, 1))
    y = models[i].predict(features)
    plt.figure(num=i)
    plt.scatter(range_x, y)
    plt.scatter(range_x, array_data[i])
    plt.show()

#Train Test split splits a numpy array into 
#data_list_variables is a list of variables, where each variable is its own numpy array of data that will be used - dependent variable is used at the end
#perc is the percent of the data we want to use
#returns a pandas dataframe of the training and test data
def TrainTestSplit(variablesDataList, perc):
    
    #this concatenates the variables you want into a pandas dataframe
    #each column is a variable
    data = variablesDataList[0]
    for i in range(1, len(variablesDataList)):
        data = pandas.concat([data, variablesDataList[i]], axis=1)

    
    data = data.sample(frac=1) #shuffles your data into training and test set. THIS MAKES YOUR MSE DIFFERENT EACH TIME

    train = data.head(round(perc*len(data.index)))
    test = data.tail(round((1-perc)*len(data.index)))

    return train, test

    #returns training data and testing data

#makes a numpy array of a given degree k from input pandas dataframe (pandas dataframe includes independent values and dependent values)
def makeFeatureMatrix(trainingData, k):
    
    poly = PolynomialFeatures(degree=k, include_bias=False)
    keys = trainingData.keys()

    FM = poly.fit_transform((trainingData[keys[0]].to_numpy()).reshape(-1, 1))

    for i in range(1,(len(keys)-1)): #takes column names


       poly_features = poly.fit_transform((trainingData[keys[i]].to_numpy()).reshape(-1, 1))
       FM = np.append(FM, poly_features, axis=1)
    
    trainingY = (trainingData[keys[len(keys)-1]].to_numpy())

    return FM, trainingY

##REMEMBER DATATYPES AND MAKE SURE THERES A ONES COLUMN IN FEATURE MATRIX

def normalize():
    normalize = 0

    return normalized

##THIS IS FOR WEATHER REGRESSION

#BROOKLYN BRIDGE


#gets brooklyn bridge data
num_people = dataset_1['Queensboro Bridge'] + dataset_1['Manhattan Bridge'] + dataset_1['Brooklyn Bridge'] + dataset_1['Williamsburg Bridge'] #number of people (dependent variable)

precipitation = dataset_1['Precipitation'] #independent
low_temp =  dataset_1['Low Temp'] #independent
high_temp = dataset_1['High Temp'] #independent

weatherList = [precipitation, low_temp, high_temp, num_people] #list of variables for training and testing split
perc = 0.9 #percent of data to become training data

trainingData, testingData = TrainTestSplit(weatherList, perc)


k = 8
#Training our model
trainFeature, trainY = makeFeatureMatrix(trainingData, k) #training set
testFeature, testY = makeFeatureMatrix(testingData, k) #testing set

poly_reg_model = LinearRegression()
poly_reg_model.fit(trainFeature, trainY)

#Testing our model
predict = poly_reg_model.predict(testFeature)
mse_test = math.mean_squared_error(testY, predict)
print(f"Test MSE with degree {k} polynomial: {mse_test}")


plt.figure(num = 'test one')
plt.scatter(range(len(testY)), testY)
plt.scatter(range(len(predict)), predict)
plt.show()
