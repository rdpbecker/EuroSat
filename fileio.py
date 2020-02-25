import csv

def writeFile(path,string):
    with open(path,"w") as f:
        f.write(string)

###############################################################
## Converts a string to a number, if possible. If the string 
## is an integer, it will be converted to an integer, if it is 
## a float it will be converted to a float, and if it is 
## neither it will be left as a string.
##
## Parameters: string - the string to be converted
##
## Returns - The input string as a string or number, depending 
##           on what it really is
###############################################################

def convertStoI(string):
    try:
        return int(string)
    except:
        pass
    try: 
        return float(string)
    except:
        return string

###############################################################
## Reads a .csv file, converting strings to numbers where 
## possible
##
## Parameters: path - the filepath of the CSV to be read
##
## Returns - The CSV as a list of lists
###############################################################

def readCsv(path):
    completeCsv = []
    with open(path) as csvFile:
        reader = csv.reader(csvFile, delimiter=',')
        for row in reader:
            completeCsv.append([convertStoI(thing) for thing in row])
    return completeCsv

###############################################################
## Writes a .csv file
##
## Parameters: path - the filepath of the CSV to be written
##             arr - the data to be written to the file
##
## Returns: None
###############################################################

def writeCsv(path,arr):
    with open(path,'w') as csvFile:
        writer = csv.writer(csvFile, delimiter=',')
        for row in arr:
            writer.writerow(row)


