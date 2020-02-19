import * from knn.py

data = readCsv('../Data/train.csv')
header = data[0]
data = removeHeader(data)
ids = [row[0] for row in data]
data = deleteNCols(data,1)
completeEnums = enumerateStrings(data)
writeCsv('../Data/train_numeric.csv',data)

test = readCsv('../Data/test.csv')
test = removeHeader(test)
test = deleteNCols(test,1)
enumerateTestData(test,completeEnums)
writeCsv('../Data/test_numeric.csv',test)

