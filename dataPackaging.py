"""
Alvin Chung
ECE 4424
This file contains a script to read in images and convert their features into a CSV file that can be used later

"""
import cv2 as cv
import numpy as np
import csv

outputPath = "data/packagedData/"

importFileType = ".jpg"
importSeqName = "data_"

exportFileType = ".jpg"
exportSeqName = "_gamma_"

acceptImagePath = "data/accepted/"

numImages = 3295

csvDataRows = list()

def getImageData(image):

    totalPixels = currentImage.shape[0] * currentImage.shape[1] #Calculate the total number of pixels in the image

    #Get the number of pixels at each intensity level
    imageHistogram = np.histogram(image, bins=256)
    counts = imageHistogram[0]

    #Convert the intensities into a probability density function
    probDensityFunc = np.divide(counts, totalPixels) 

    return list(probDensityFunc)

#Returns and writes out three gamma adjusted images
#With gamma values 2.0, 3.0, and 4.0
def getGammaImages(image, imageNum):
    #Convert to [0,1] intensity range
    zeroOneRangeImage = np.divide(image, 255)

    gammaValA = 2.0
    gammaValB = 3.0
    gammaValC = 4.0

    #Perform gamma adjustment
    gammaImageA = np.power(zeroOneRangeImage, gammaValA)
    gammaImageB = np.power(zeroOneRangeImage, gammaValB)
    gammaImageC= np.power(zeroOneRangeImage, gammaValC)

    #Convert back to 8-bit range
    gammaImageA = np.multiply(gammaImageA, 255)
    gammaImageB = np.multiply(gammaImageB, 255)
    gammaImageC = np.multiply(gammaImageC, 255)

    gammaImageA = gammaImageA.astype(dtype=np.uint8)
    gammaImageB = gammaImageB.astype(dtype=np.uint8)
    gammaImageC = gammaImageC.astype(dtype=np.uint8)

    outputA = outputPath + importSeqName + str(imageNum) + exportSeqName + "A" + exportFileType
    outputB = outputPath + importSeqName + str(imageNum) + exportSeqName + "B" + exportFileType
    outputC = outputPath + importSeqName + str(imageNum) + exportSeqName + "C" + exportFileType

    dataA = getImageData(gammaImageA)
    dataB = getImageData(gammaImageB)
    dataC = getImageData(gammaImageC)

    dataA.insert(0, importSeqName + str(imageNum) + exportSeqName + "A")
    dataB.insert(0, importSeqName + str(imageNum) + exportSeqName + "B")
    dataC.insert(0, importSeqName + str(imageNum) + exportSeqName + "C")

    csvDataRows.append(dataA)
    csvDataRows.append(dataB)
    csvDataRows.append(dataC)

    cv.imwrite(outputA, gammaImageA)
    cv.imwrite(outputB, gammaImageB)
    cv.imwrite(outputC, gammaImageC)


for imageNum in range(0, numImages):

    #Form the image path
    currentImagePath = acceptImagePath + importSeqName + str(imageNum) + importFileType
    currentImage = cv.imread(currentImagePath, cv.IMREAD_GRAYSCALE)

    #Form the output image
    outputPathFinal = outputPath + importSeqName + str(imageNum) + importFileType

    currentData = getImageData(currentImage)
    currentData.insert(0, (importSeqName +  str(imageNum)))

    csvDataRows.append(currentData)

    cv.imwrite(outputPathFinal, currentImage)
    getGammaImages(currentImage, imageNum)

    if(imageNum % 10 == 0):
        print("Processed " + str(imageNum) + "/" + str(numImages) + " images")

file = open('rawData.csv', 'w', newline='')

writer = csv.writer(file)

#Creating the field
field = ["Name"] 

intLevels = list(range(0, 256))
namedLevels = list()
for level in intLevels:
    thisLevel = "Intensity "
    thisLevel += str(level)
    field.append(thisLevel)

writer.writerow(field)

for row in csvDataRows:
    writer.writerow(row)

file.close()