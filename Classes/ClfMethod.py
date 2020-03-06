import sys

def validateMethod(method):
    if not method in ['knn','bayes','forest','svm']:
        print("Invalid learning method")
        sys.exit()

class ClfMethod:
    clf = None
    method = None

    def __init__(self,clf,method):
        self.clf = clf
        validateMethod(method)
        self.method = method
