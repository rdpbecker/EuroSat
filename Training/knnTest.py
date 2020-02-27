from sklearn.neighbors import KNeighborsClassifier
import timeit
import sys
sys.path.append("../")
sys.path.append("../DataProcessing")
import fileio
import dataManip

start = timeit.default_timer()
data = fileio.readCsv('../Data/Training/train_numeric.csv')
testData = fileio.readCsv('../Data/Testing/test_numeric.csv')
testOriginal = fileio.readCsv('../Data/Testing/test.csv')
print("Reading time: ", timeit.default_timer()-start)

ids = dataManip.firstColVector(testOriginal)
ids = ids[1:]

start = timeit.default_timer()
out = dataManip.separateCol(data,-1)
data = out[0]
classes = dataManip.firstColVector(out[1])
print("Separation time: ", timeit.default_timer()-start)

start = timeit.default_timer()
neigh = KNeighborsClassifier(n_neighbors=30)
neigh.fit(data,classes)
print("Training time: ",timeit.default_timer()-start)

output = []
start = timeit.default_timer()
for i in range(len(testData)):
    output.append([ids[i],neigh.predict([data[i]])[0]])

print("Predicting time: ",timeit.default_timer()-start)
fileio.writeCsv("../Output/knn_30.csv",output)
