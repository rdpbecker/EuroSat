def createTable(table,clf,data,classes):
    for i in range(len(data)):
        if not i%250-1:
            print("Classifying observation number: ",i)
        table.addObs(classes[i],clf.predict([data[i]])[0])
