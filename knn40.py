from sklearn.neighbors import KNeighborsClassifier
import timeit
import knn

start = timeit.default_timer()
data = knn.readCsv('../Data/train_numeric.csv')
testData = knn.readCsv('../Data/test_numeric.csv')
print("Reading time: ", timeit.default_timer()-start)

start = timeit.default_timer()
out = knn.separateCol(data,-1)
data = out[0]
classes = knn.firstColVector(out[1])
print("Separation time: ", timeit.default_timer()-start)

start = timeit.default_timer()
neigh = KNeighborsClassifier(n_neighbors=40)
neigh.fit(data,classes)
print("Training time: ",timeit.default_timer()-start)

output = []
start = timeit.default_timer()
for i in range(len(testData)):
    output.append([neigh.predict([data[i]])[0]])

print("Predicting time: ",timeit.default_timer()-start)
knn.writeCsv("../Output/knn_40.csv",output)
