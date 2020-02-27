import random

###############################################################
## Remove the first row of an array (list of lists)
##
## Parameters: arr - the array to delete the header of
##
## Returns - The original array without the header (first row)
###############################################################

def removeHeader(arr):
    return arr[1:]

###############################################################
## Separates an array by columns into two blocks. All the data 
## will be included in one of the two parts, and none will be 
## duplicated.
##
## Parameters: arr - the array to be split
##             col - the first column to be included in the 
##                   right part
##
## Returns - A list with two elements. The first element is 
##           the left part of the array, and the second element 
##           is the right part of the array
###############################################################

def separateCol(arr,col):
    n = len(arr)
    return [[arr[i][:col] for i in range(n)], [arr[i][col:] for i in range(n)]]

###############################################################
## Deletes the first few columns of an array
##
## Parameters: arr - the array to delete columns from
##             col - the number of rows to be deleted
##
## Returns - A copy of the array with the first "col" columns 
##           deleted
###############################################################

def deleteNCols(arr,col):
    return [arr[i][col:] for i in range(len(arr))]

def deleteL(arr):
    return deleteNCols(removeHeader(arr),1)

###############################################################
## Creates a vector from the first column of an array
##
## Parameters: arr - the array to create the vector from
##
## Returns - The elements of the first column of the array as a 
##           list
###############################################################

def firstColVector(arr):
    return [arr[i][0] for i in range(len(arr))]

###############################################################
## Splits a dataset into training and test data by a split of 
## roughly 75-25
##
## Parameters: data - the data to be split
## 
## Returns: A list of lists - [trainData, testData]
###############################################################

def splitSamples(data):
    random.seed(1)
    random.shuffle(data)
    length = len(data)
    cutoff = int(0.2*length*random.random()+0.74*length)
    return [data[:cutoff],data[cutoff:]]


