"""
Alvin Chung
ECE 4424
This file contains a script to read rawData from a CSV file and extract features from the data
then it can create, train, and test a model

"""
import math
import csv
import sklearn
import sklearn.linear_model
import sklearn.multioutput
import numpy as np
import cv2 as cv

truthValues = list()
featureLists = list()
dataDictionary = dict()


def getMSE(observed, predicted):
    
    obvArr = np.array(observed)
    predArr = np.array(predicted)

    diff = np.subtract(obvArr, predArr)

    squared = np.power(diff, 2)

    mseSum = np.sum(squared)

    result = mseSum / len(observed)

    return result

#Since the data taken directly from the CSV file are strings, have to convert them to actual values
def convertToNumList(thisList):
    actualValues = list()

    for value in thisList:
        actualValues.append(float(value))

    return actualValues

#Returns a list of features, designed by the user
def getFeatures(values):

    actualValues = values

    newFeatures = list()

    #Currently, getting the total probability
    newFeatures.append(sum(actualValues[0:32]))
    newFeatures.append(sum(actualValues[32:64]))
    newFeatures.append(sum(actualValues[64:96]))
    newFeatures.append(sum(actualValues[96:128]))
    newFeatures.append(sum(actualValues[128:160]))
    newFeatures.append(sum(actualValues[160:192]))
    newFeatures.append(sum(actualValues[192:224]))
    newFeatures.append(sum(actualValues[224:]))

    return newFeatures

#Read data from CSV file
file = open("rawData.csv", "r")
reader = csv.reader(file)

#Storing data in a dictionary
for line in reader:
    key = line[0]
    value = line[1:]

    #Convert to actual numbers
    if(key == "Name"):
        continue

    value = convertToNumList(value)

    dataDictionary[key] = value

file.close()

#Train on half of the data
keys = list(dataDictionary.keys())
splitIndex = math.floor(len(keys) / 2)
trainKeys = keys[:splitIndex]
testKeys= keys[splitIndex:]

#Get the features and truth values
for key in trainKeys:

    if(key == "Name"):
        continue

    #Select the gamma adjusted data to use for training
    if( not ("gamma" in key)):
        continue
    
    #Getting the ground truth data
    gammaNameIndex = key.index("_gamma")
    truthName = key[0:gammaNameIndex]

    values = dataDictionary[key]
    theseFeatures = getFeatures(values)

    truthValues.append(dataDictionary[truthName]) #Ground truth
    featureLists.append(theseFeatures)            #Features


#Stochastic Gradient Ascent Regressor
sgdReg = sklearn.linear_model.SGDRegressor(max_iter=50, tol=0.05)

#Multivariate
multivarReg = sklearn.multioutput.MultiOutputRegressor(sgdReg)

print("FITTING DATA")
#Fit to data
multivarReg.fit(featureLists, truthValues)

x = dataDictionary["data_1032_gamma_B"]
y1 = dataDictionary["data_1032"]

alpha = getFeatures(x)

y = multivarReg.predict([alpha])

mseList = list()
mseListNorm = list()

print("EVALUATING TRAINING DATA")
#Run through training data again
#Get the features and truth values
for key in trainKeys:

    if(key == "Name"):
        continue

    #Select the gamma adjusted data to use for training
    if( not ("gamma" in key)):
        continue
    
    #Getting the ground truth data
    gammaNameIndex = key.index("_gamma")
    truthName = key[0:gammaNameIndex]
    groundTruth = dataDictionary[truthName]

    #Get features for this x
    values = dataDictionary[key]
    theseFeatures = getFeatures(values)
    
    prediction = multivarReg.predict([theseFeatures])
    
    #Manually normalized version
    predSum = np.sum(prediction)
    predictionNorm = np.divide(prediction, predSum)

    thisMSE = getMSE(prediction, groundTruth)
    thisMSENorm = getMSE(predictionNorm, groundTruth)

    mseList.append(thisMSE)
    mseListNorm.append(thisMSENorm)

avgMSE = sum(mseList) / len(mseList)
avgMSENorm = sum(mseListNorm) / len(mseListNorm)

print("Train Data Average Mean-Squared Error: ", avgMSE)
print("Train Data Average Mean-Squared Error (Manually Normalized): ", avgMSENorm)


mseList = list()
mseListNorm = list()

print()
print("EVALUATING TESTING DATA")

#Run through test data
for keys in testKeys:
    if(key == "Name"):
        continue

    #Select the gamma adjusted data
    if( not ("gamma" in key)):
        continue
    
    #Getting the ground truth data
    gammaNameIndex = key.index("_gamma")
    truthName = key[0:gammaNameIndex]
    groundTruth = dataDictionary[truthName]

    #Get features for this x
    values = dataDictionary[key]
    theseFeatures = getFeatures(values)

    prediction = multivarReg.predict([theseFeatures])

    #Manually normalized version
    predSum = np.sum(prediction)
    predictionNorm = np.divide(prediction, predSum)

    thisMSE = getMSE(prediction, groundTruth)
    thisMSENorm = getMSE(predictionNorm, groundTruth)

    mseList.append(thisMSE)
    mseListNorm.append(thisMSENorm)

avgMSE = sum(mseList) / len(mseList)
avgMSENorm = sum(mseListNorm) / len(mseListNorm)

print("Test Data Average Mean-Squared Error: ", avgMSE)
print("Test Data Average Mean-Squared Error (Manually Normalized): ", avgMSENorm)

