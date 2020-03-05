import sys
sys.path.append("../")
sys.path.append("../Classes")
import fileio
import Voters

def createPaths(files):
    paths = []
    for f in files:
        paths.append("../Output/"+f+".csv")
    return paths

def getIds(path):
    arr = fileio.readCsv(path)
    ids = []
    for i in range(1,len(arr)):
        ids.append(arr[i][0])
    return ids

def main(outpath,paths):
    voters = Voters.Voters()
    for path in paths:
        voters.addVoter(fileio.readCsv(path))

    ids = getIds(paths[0])
    output = [["Ids","Predicted"]]
    for ID in ids:
        output.append([ID,voters.vote(ID)])

    fileio.writeCsv(outpath+".csv",output)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Need an output target and an input predictor")
        sys.exit()
    outpath = sys.argv[1]
    paths = createPaths(sys.argv[2:])
    main(outpath,paths)
