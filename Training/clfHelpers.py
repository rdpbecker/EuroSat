def createTable(table,clf,data,classes):
    for i in range(len(data)):
        if not i%250-1:
            print("Classifying observation number: ",i)
        table.addObs(classes[i],clf.predict([data[i]])[0])

def createEnsembleTable(table,voters,datas,classes):
    for i in range(len(datas[0])):
        if not i%250-1:
            print("Classifying observation number: ",i)
        table.addObs(classes[0][i],voters.vote([datas[0][i]]))
#[datas[j][i] for j in range(len(datas))]

def createTableWithArr(table,arr,clf,data,classes,ids):
    for i in range(len(data)):
        if not i%250-1:
            print("Classifying observation number: ",i)
        prediction = clf.predict([data[i]])[0]
        arr.append([ids[i],prediction])
        table.addObs(classes[i],prediction)
