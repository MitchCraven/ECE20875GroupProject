import pandas
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as math
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing as pros
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn import metrics
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


def get_day_modle(para):
    # Split
    hl_sizes, rand_state, act_func = para

    # Make modle
    model = MLPClassifier(hidden_layer_sizes=hl_sizes, random_state=rand_state, activation=act_func)

    return model
def normalize_train(X_train):

    # fill in
    array = np.array(X_train).T
    mean = [0] * len(array)
    std = [0] * len(array)

    for i in range(len(array)):
        mean[i] = np.mean(array[i])
        std[i] = np.std(array[i])

    sub_mean = (np.subtract(array.T, mean))
    X = np.divide(sub_mean, std)
    return X, mean, std
def normalize_test(X_test, trn_mean, trn_std):

    # fill in
    X = np.divide(np.subtract(X_test, trn_mean), trn_std)
    return X
def conf_matrix(y_pred, y_true, num_class):
    """
    agrs:
    y_pred : List of predicted classes
    y_true : List of cooresponding true class labels
    num_class : The number of distinct classes being predicted

    Returns:
    M : Confusion matrix as a numpy array with dimensions (num_class, num_class)
    """
    # Your code here. We ask that you not use an external libary like sklearn to create the confusion matrix and code this function manually
    out = np.zeros((num_class, num_class))
    tot = [0] * num_class
    no = [0] * num_class
    for i in range(0, num_class):
        for j in range(len(y_true)):
            if (y_pred[j] == y_true[j]) & (y_true[j] == str(i)):
                tot[i] += 1
                out[i, i] += 1
            elif (y_pred[j] != y_true[j]) & (y_true[j] == str(i)):
                no[i] += 1
                val = int(y_pred[j])
                out[val, i] += 1
    return out

#Number of bridges people regression for sensors
b = dataset_1['Brooklyn Bridge'] #brooklyn bridge numbers
m = dataset_1['Manhattan Bridge'] #manhattan bridge numbers
q = dataset_1['Queensboro Bridge'] #queensboro bridge numbers
w = dataset_1['Williamsburg Bridge'] #williamsburg bridge numbers

#these are y values
dataset_1['Total'] = b + m + q + w
total = dataset_1['Total']

#NORMALIZATION BASED ON TOTAL
mean = total.mean()
stdev = total.std()
total = (total - mean)/stdev
b = (b - mean)/stdev
m = (m - mean)/stdev
q = (q - mean)/stdev
w = (w - mean)/stdev



perc = 0.8
for i in [[b, m, q, total], [b, m, w, total], [b, q, w, total], [b, q, w, total]]:
    #b m q sensors
    trainingData, testingData = TrainTestSplit(i, perc)

    #training the model
    poly, trainY = makeFeatureMatrix(trainingData, 1)
    poly_reg_model = LinearRegression()
    poly_reg_model.fit(poly_features, trainY)


    test_p, testY = makeFeatureMatrix(testingData, 1)
    predict = poly_reg_model.predict(test_features)
    mse_test =  math.mean_squared_error(testY, predict, squared=False)

    print(f"MSE of [{i[0].keys()}, {i[1].keys()}, {i[2].keys()}] : {mse_test}")







##THIS IS FOR WEATHER REGRESSION

#BROOKLYN BRIDGE


#gets brooklyn bridge data

num_people = dataset_1['Manhattan Bridge'] + dataset_1['Manhattan Bridge'] + dataset_1['Brooklyn Bridge'] + dataset_1['Williamsburg Bridge'] #number of people (dependent variable)
precipitation = dataset_1['Precipitation'] #independent
low_temp =  dataset_1['Low Temp'] #independent
high_temp = dataset_1['High Temp'] #independent

weatherList = [precipitation, low_temp, high_temp, num_people] #list of variables for training and testing split
perc = 0.8 #percent of data to become training data

trainingData, testingData = TrainTestSplit(weatherList, perc)

past = 0
for k in range(1, 20):
    #Training our model
    trainFeature, trainY = makeFeatureMatrix(trainingData, k) #training set

    testFeature, testY = makeFeatureMatrix(testingData, k) #testing set
    #trainFeature, mean, std = normalize_train(trainFeature)
    #testFeature = normalize_test(testFeature, mean, std)

    poly_reg_model = LinearRegression()
    poly_reg_model.fit(trainFeature, trainY)

    #Testing our model
    predict = poly_reg_model.predict(testFeature)
    r2 = r2_score(testY, predict)
    if r2 > past:
        past = r2
        deg = k

print(f"Test MSE with degree {deg} polynomial: {r2}")


plt.figure(num='test one')
plt.scatter(range(len(testY)), testY)
plt.scatter(range(len(predict)), predict)
plt.show()

#Days Question
#shuff = dataset_1.sample(frac=1)
y = dataset_1['Day']
x = dataset_1['Manhattan Bridge']

params = [(10, 15, 7), 1, "relu"]
mod = get_day_modle(params)

# Data Split
trainX = x.head(round(perc*len(x.index)))
testX = x.tail(round((1-perc)*len(x.index)))
trainY = y.head(round(perc*len(y.index)))
testY = y.tail(round((1-perc)*len(y.index)))


# Reshape
trainY = trainY.to_numpy().reshape(-1, 1)
trainX = trainX.to_numpy().reshape(-1, 1)
testY = testY.to_numpy().reshape(-1, 1)
testX = testX.to_numpy().reshape(-1, 1)

# Normalize
gen_trainX, mean, std = normalize_train(trainX)
gen_testX = normalize_test(testX, mean, std)

#Modle
mod.fit(gen_trainX, trainY.ravel())
predict = mod.predict(gen_testX)



holder = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
for i in range(0, 7):
    trainY[trainY == holder[i]] = i
    testY[testY == holder[i]] = i
    predict[predict == holder[i]] = i
pred_int = [int(p) for p in predict]
acc = metrics.accuracy_score(testY.tolist(), pred_int)
# 5. Calculate the confusion matrix by using the completed the function above
conf_mat = conf_matrix(testY, predict, 7)

# 6. Compute the AUROC score. You may use metrics.roc_auc_score(...)
y_score = mod.predict_proba(testX)
auc_score = metrics.roc_auc_score(testY.tolist(), y_score, multi_class='ovr')
print(auc_score)
print(acc)


