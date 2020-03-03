from sklearn.ensemble import RandomForestClassifier 
import timeit
import sys
sys.path.append("../")
sys.path.append("../DataProcessing")
import fileio
import dataManip

start = timeit.default_timer()
data = fileio.readCsv('../Data/Training/train_processed.csv',nerf=True)
testData = fileio.readCsv('../Data/Testing/test_processed.csv',nerf=True)
testOriginal = fileio.readCsv('../Data/Testing/test.csv')
print("Reading time: ", timeit.default_timer()-start)

ids = dataManip.firstColVector(testOriginal)
ids = ids[1:]

data = dataManip.deleteL(data)
testData = dataManip.deleteL(testData)

start = timeit.default_timer()
out = dataManip.separateCol(data,-1)
data = out[0]
classes = dataManip.firstColVector(out[1])
print("Separation time: ", timeit.default_timer()-start)

start = timeit.default_timer()
clf = RandomForestClassifier(\
    n_estimators=1000,\
    max_depth=15,\
    min_samples_leaf=10,\
    ccp_alpha=0\
)
clf.fit(data,classes)
print("Training time: ",timeit.default_timer()-start)

output = [['ids','predicted']]
start = timeit.default_timer()
for i in range(len(testData)):
    output.append([ids[i],clf.predict([data[i]])[0]])

print("Predicting time: ",timeit.default_timer()-start)
fileio.writeCsv("../Output/forest_1000_15_10_0.csv",output)
