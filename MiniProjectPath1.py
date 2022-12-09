import pandas
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as math
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import r2_score, roc_curve
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
''' 
The following is the starting code for path1 for data reading to make your first step easier.
'dataset_1' is the clean data for path1.
'''
# Data In
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
    
    FM = np.append(FM, np.ones((len(FM), 1)), axis=1)
    trainingY = (trainingData[keys[len(keys)-1]].to_numpy()).T

    return FM, trainingY

##REMEMBER DATATYPES AND MAKE SURE THERES A ONES COLUMN IN FEATURE MATRIX

# Gets the MPLC, GNB and SVC Models
def get_day_modle(para):
    # Split
    hl_sizes, rand_state, act_func = para

    # Make modle
    model_ner = MLPClassifier(hidden_layer_sizes=hl_sizes, random_state=rand_state, activation=act_func)
    model_GNB = GaussianNB()
    model_SVC = SVC(random_state=1, probability=True)
    return model_ner, model_GNB, model_SVC

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

# Normalizes the Test Data Based on the Test Data
def normalize_test(X_test, trn_mean, trn_std):

    # fill in
    X = np.divide(np.subtract(X_test, trn_mean), trn_std)
    return X

#Makes the conf Matrix
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


print('\n PART 1')
perc = 0.8

for i in [[b, m, q, total], [b, m, w, total], [b, q, w, total], [b, q, w, total]]:
    #b m q sensors
    trainingData, testingData = TrainTestSplit(i, perc)

    #training the model
    poly, trainY = makeFeatureMatrix(trainingData, 1)
    poly_reg_coeffs = np.linalg.inv(poly.T @ poly) @ poly.T @ trainY

    test_p, testY = makeFeatureMatrix(testingData, 1)
    predict = test_p @ poly_reg_coeffs
    mse_test =  math.mean_squared_error(testY, predict, squared=False)

    print(f"MSE of [{i[0].name}, {i[1].name}, {i[2].name}] : {mse_test}")
    
    #we plotted the three bridges we were putting sensors on
    #plot first bridge
    plt.title(f"Model with Sensors on {i[0].name}, {i[1].name}, {i[2].name}", fontsize = 16)
    plt.subplot(311)
    plt.scatter(((testingData[i[0].name]*stdev)+mean), ((testY)*stdev+mean), color="Black", label=f'Total vs {i[0].name}')
    plt.scatter(((testingData[i[0].name]*stdev)+mean), ((predict)*stdev+mean), color="Red", label=f'Predicted Total vs {i[0].name}')
    plt.ylabel('Total People', fontsize=12)
    plt.xlabel(f'{i[0].name} People', fontsize=12)
    plt.legend(fontsize=10, loc='upper left')

    #pot second bridge
    plt.subplot(312)
    plt.scatter(((testingData[i[1].name]*stdev)+mean), ((testY)*stdev+mean), color="Black", label=f'Total vs {i[1].name}')
    plt.scatter(((testingData[i[1].name]*stdev)+mean), ((predict)*stdev+mean), color="Red", label=f'Predicted Total vs {i[1].name}')
    plt.ylabel('Total People', fontsize=12)
    plt.xlabel(f'{i[1].name} People', fontsize=12)
    plt.legend(fontsize=10, loc='upper left')

    #Plot third bridge
    plt.subplot(313)
    plt.scatter(((testingData[i[2].name]*stdev)+mean), ((testY)*stdev+mean), color="Black", label=f'Total vs {i[2].name}')
    plt.scatter(((testingData[i[2].name]*stdev)+mean), ((predict)*stdev+mean), color="Red", label=f'Predicted Total vs {i[2].name}')
    plt.ylabel('Total People', fontsize=12)
    plt.xlabel(f'{i[2].name} People', fontsize=12)
    plt.legend(fontsize=10, loc='upper left')
    plt.show()

##THIS IS FOR WEATHER REGRESSION

#BROOKLYN BRIDGE


#gets brooklyn bridge data

num_people = dataset_1['Manhattan Bridge'] + dataset_1['Manhattan Bridge'] + dataset_1['Brooklyn Bridge'] + dataset_1['Williamsburg Bridge'] #number of people (dependent variable)
precipitation = dataset_1['Precipitation'] #independent
low_temp =  dataset_1['Low Temp'] #independent
high_temp = dataset_1['High Temp'] #independent

#normalizes all data
npMean = num_people.mean()
npStd = num_people.std()
num_people = (num_people-num_people.mean())/num_people.std()

precMean = precipitation.mean()
precStd = precipitation.std()
precipitation = (precipitation-precipitation.mean())/precipitation.std()

ltMean = low_temp.mean()
ltStd = low_temp.std()
low_temp = (low_temp-low_temp.mean())/low_temp.std()

htMean = high_temp.mean()
htStd = high_temp.std()
high_temp = (high_temp-high_temp.mean())/high_temp.std()


weatherList = [precipitation, low_temp, high_temp, num_people] #list of variables for training and testing split
perc = 0.8 #percent of data to become training data

print('\n PART 2')
trainingData, testingData = TrainTestSplit(weatherList, perc)

past = 0
for k in range(1, 9):
    #Training our model
    poly, trainY = makeFeatureMatrix(trainingData, k) #training set

    test_p, testY = makeFeatureMatrix(testingData, k) #testing set
    #trainFeature, mean, std = normalize_train(trainFeature)
    #testFeature = normalize_test(testFeature, mean, std)

    poly_reg_coeffs = np.linalg.inv(poly.T @ poly) @ poly.T @ trainY

    #make predictions
    predict = test_p @ poly_reg_coeffs
    mse_test =  math.mean_squared_error(testY, predict, squared=False)
    print(f"Test MSE with degree {k} polynomial: {mse_test}")

    #we plotted the three bridges we were putting sensors on
    #plot Precipitation
    plt.title(f"Precipitation, High Temp, Low Temp, vs Total", fontsize = 20)
    plt.subplot(311)
    plt.scatter(((testingData['Precipitation']*precStd)+precMean), ((testY*npStd)+npMean), color="Black", label=f'Total vs Precipitation')
    plt.scatter(((testingData['Precipitation']*precStd)+precMean), ((predict*npStd)+npMean), color="blue", label=f'Predicted Total vs Precipitation')
    plt.xlabel('Precipitation', fontsize=12)
    plt.ylabel(f'Total People', fontsize=12)
    plt.legend(fontsize=10, loc='upper left')

    #pot high temp
    plt.subplot(312)
    plt.scatter(((testingData['High Temp']*htStd)+htMean), ((testY*npStd)+npMean), color="Black", label=f'Total vs High Temp')
    plt.scatter(((testingData['High Temp']*htStd)+htMean), ((predict*npStd)+npMean), color="blue", label=f'Predicted Total vs High Temp')
    plt.xlabel('High Temp', fontsize=12)
    plt.ylabel(f'Total People', fontsize=12)
    plt.legend(fontsize=10, loc='upper left')

    #Plot low temp
    plt.subplot(313)
    plt.scatter(((testingData['Low Temp']*ltStd)+ltMean), ((testY*npStd)+npMean), color="Black", label=f'Total vs Low Temp')
    plt.scatter(((testingData['Low Temp']*ltStd)+ltMean), ((predict*npStd)+npMean), color="blue", label=f'Predicted Total vs Low Temp')
    plt.xlabel('Low Temp', fontsize=12)
    plt.ylabel(f'Total People', fontsize=12)
    plt.legend(fontsize=10, loc='upper left')
    plt.show()




    #Testing our model
    r2 = r2_score(testY, predict)
    if r2 > past:
        past = r2
        deg = k

print(f"R squared with degree {deg} polynomial: {r2}")







# Days Question 3
print("Part 3")
# Used to Iterate Through All the Bridges
titl = ['Brooklyn Bridge', 'Manhattan Bridge', 'Queensboro Bridge', 'Williamsburg Bridge']
for k in titl:

    # Preparing Data
    shuff = dataset_1.sample(frac=1)
    y = shuff['Day']
    x = shuff[k]

    params = [(10, 15, 7), 1, "relu"]
    model_ner, model_GNB, model_SVC = get_day_modle(params)

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
    #gen_trainX, mean, std = normalize_train(trainX)
    #gen_testX = normalize_test(testX, mean, std)

    # Neural Model
    model_ner.fit(trainX, trainY.ravel())
    predict_ner = model_ner.predict(testX)

    # GNB Model
    model_GNB.fit(trainX, trainY.ravel())
    predict_GNB = model_GNB.predict(testX)

    # SVC Model
    model_SVC.fit(trainX, trainY.ravel())
    predict_SVC = model_SVC.predict(testX)

    # Accuracy Score of Each Model
    acc_ner = metrics.accuracy_score(testY.tolist(), predict_ner)
    acc_GNB = metrics.accuracy_score(testY.tolist(), predict_GNB)
    acc_SVC = metrics.accuracy_score(testY.tolist(), predict_SVC)

    # Confusion Matrix
    conf_mat_ner = conf_matrix(testY, predict_ner, 7)
    conf_mat_GNB = conf_matrix(testY, predict_GNB, 7)
    conf_mat_SVC = conf_matrix(testY, predict_SVC, 7)

    # AUC Neural
    y_score_ner = model_ner.predict_proba(testX)
    auc_score_ner = metrics.roc_auc_score(testY.tolist(), y_score_ner, multi_class='ovr')

    # AUC GNB
    y_score_GNB = model_GNB.predict_proba(testX)
    auc_score_GNB = metrics.roc_auc_score(testY.tolist(), y_score_GNB, multi_class='ovr')

    # AUC SVC
    y_score_SVC = model_SVC.predict_proba(testX)
    auc_score_SVC = metrics.roc_auc_score(testY.tolist(), y_score_SVC, multi_class='ovr')

    # Graph AUROC
    #false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(testY, y_score_GNB)

    # Prints Results of Neural Network
    print(k + " AUC ROC Score Multi-Layer Neural: " + str(auc_score_ner))
    print(k + " Accuracy Multi-layer Nural: " + str(acc_ner))
    print()

    # Prints Results of GNB
    print(k + " AUC ROC Score GNB: " + str(auc_score_GNB))
    print(k + " Accuracy GNB: " + str(acc_GNB))
    print()

    # Prints Results of SVC
    print(k + " AUC ROC Score SVC: " + str(auc_score_SVC))
    print(k + " Accuracy SVC: " + str(acc_SVC))
    print()



##MISC## Or Unused
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