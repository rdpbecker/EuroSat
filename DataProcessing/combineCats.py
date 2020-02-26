import sys
sys.path.append("../")
import fileio

def main(path):
    data = fileio.readCsv(path+".csv")
    cats = fileio.readCsv("../Data/satisfied.csv")
    for i in range(len(data)):
        data[i].append(cats[i][0])
    fileio.writeCsv(path+"_COPY.csv",data)

if __name__ == "__main__":
    path = "../Data/" + sys.argv[1] + "/" + sys.argv[2]
    main(path)
