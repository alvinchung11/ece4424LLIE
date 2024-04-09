import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

#Convert Images to grayscale

numImages = 2792

importFileType = ".jpg"
importSeqName = "image"

exportFileType = ".jpg"
exportRejectSeqName = "reject_"
exportAcceptSeqName = "data_"

originalImagePath = "data/original/"
rejectImagePath = "data/rejected/"
acceptImagePath = "data/accepted/"

#If the sum of the probabilities for the shadow or highlight intensities exceed this value,
#the image is rejected for use
#Essentially, the lower the number, the more strict it is
rejectThreshold = 0.45
rejectThresholdDev = 0.05 #The max deviation allowed from the threshold

#Defining the range of shadows, mids, and highlights
midRangeBeginIndex = 85 #+ 20 
highlightBeginIndex = 171 #+ 50

acceptCount = 0
rejectCount = 0

for imageNum in range(1, numImages + 1):

    #Form the image path
    currentImagePath = originalImagePath + importSeqName + " (" + str(imageNum) + ")" + importFileType

    #Read in the image as a grayscale image
    currentImage = cv.imread(currentImagePath, cv.IMREAD_GRAYSCALE)

    #Some images may be corrupt, so account for that will this try except
    try:
        totalPixels = currentImage.shape[0] * currentImage.shape[1] #Calculate the total number of pixels in the image
    except AttributeError:
        continue

    #Get the number of pixels at each intensity level
    imageHistogram = np.histogram(currentImage, bins=256)
    counts = imageHistogram[0]

    #Convert the intensities into a probability density function
    probDensityFunc = np.divide(counts, totalPixels) 

    #Get the probability values for intensities of the shadows and highlights
    shadows = probDensityFunc[0:midRangeBeginIndex]
    #midRange = probDensityFunc[midRangeBeginIndex:highlightBeginIndex]
    highlights = probDensityFunc[highlightBeginIndex:]

    #Calculate the sum of all the probability values in the shadows and highlights
    shadowProb = shadows.sum()
    highlightProb = highlights.sum()

    shadowHighlightProb = shadowProb + highlightProb

    #If the total probability of a shadow or hightlight intensity exceeds the threshold,
    #reject the image for use for training and testing

    if(abs(rejectThreshold - shadowHighlightProb) > rejectThresholdDev):
        exportImagePath = rejectImagePath + exportRejectSeqName + str(rejectCount) + exportFileType
        rejectCount += 1
    else:
        exportImagePath = acceptImagePath + exportAcceptSeqName + str(acceptCount) + exportFileType
        acceptCount += 1

    #print(len(probDensityFunc), len(shadows), len(midRange), len(highlights))

    cv.imwrite(exportImagePath, currentImage)

    if(imageNum % 10 == 0):
        print("Processed " + str(imageNum) + "/" + str(numImages) + " images")

print("All Images Processed")
