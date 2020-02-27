import sys
sys.path.append("../")
sys.path.append("../DataProcessing")
from fileio import *
from dataManip import *
from enumeration import *

data = readCsv('../Data/Training/train.csv')
header = data[0]
data = removeHeader(data)
ids = [row[0] for row in data]
data = deleteNCols(data,1)
completeEnums = enumerateStrings(data)
#writeCsv('../Data/Training/train_numeric.csv',data)

test = readCsv('../Data/Testing/test.csv')
test = removeHeader(test)
test = deleteNCols(test,1)
enumerateTestData(test,completeEnums)
#writeCsv('../Data/Testing/test_numeric.csv',test)

