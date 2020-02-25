import sys
sys.path.append("../Classes")
import numpy as np
import CompleteEnum from CompleteEnum

###############################################################
## Checks if there are strings in a specified column of an 
## array
##
## Parameters: arr - the array to be checked for strings
##             col - the column to be checked
##
## Returns - If there are strings in the column, this returns a 
##           list of two lists. The first list is a list of the 
##           unique strings in the column, and the second is a 
##           list of the unique numbers in the list. If there 
##           are no strings in the column, this returns None.
###############################################################

def checkString(arr,col):
    strings = []
    nums = []
    flag = False
    for i in range(len(arr)):
        thing = arr[i][col]
        if isinstance(thing,str):
            flag = True
            strings.append(thing)
        else:
            nums.append(thing)
    if flag:
        return [np.unique(strings).tolist(),np.unique(nums).tolist()]
    return None 

###############################################################
## Takes a set of strings and enumerates them, avoiding certain
## values
##
## Parameters: strings - a list of strings to be enumerated
##             ints - a list of disallowed integers
##
## Returns - A dictionary whose keys are the input strings and 
##           whose values are unique integers, none of which 
##           are contained in the list of integers.
###############################################################

def genDict(strings, ints):
    count = 0
    enum = {}
    while len(strings):
        if not count in ints:
            enum[strings[0]] = count
            strings.pop(0)
        count = count + 1
    return enum

###############################################################
## Converts all the strings in a specified column to integers 
## based on a given enumeration of those strings
##
## Parameters: arr - the array to be converted
##             col - the column to be converted 
##             enum - an enumeration of the strings in the column
##
## Returns - None - the conversion is done in place
###############################################################

def adjust(arr,col,enum):
    for i in range(len(arr)):
        if isinstance(arr[i][col],str):
            arr[i][col] = enum[arr[i][col]]

###############################################################
## Iterates through the columns of an array, converting strings 
## to integers and ensuring that the strings in a column are 
## converted to a number that is not already present in that 
## column
##
## Parameters: arr - the array to be converted
##
## Returns - A dictionary in which the keys are the columns
##           that were adjusted and the values are the 
##           enumerations for the columns. We return these so
##           that we use the same enumerations for the test
##           data.
###############################################################

def enumerateStrings(arr):
    completeEnums = {}
    for col in range(len(arr[0])):
        check = checkString(arr,col)
        if not check is None:
            enum = genDict(check[0],check[1])
            completeEnums[col] = CompleteEnum(enum,check[1])
            adjust(arr,col,enum)
    return completeEnums

###############################################################
## Enumerate the test data using the same enumerations as for 
## the training data.
##
## Parameters: arr - the array to be enumerated
##             enums - the enumerations used for the training
##                     data
##

