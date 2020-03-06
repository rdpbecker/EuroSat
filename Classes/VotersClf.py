import random
import ClfMethod

dataChoice = {'bayes': 0, 'forest': 0, 'knn': 1, 'svm': 1}

class VotersClf:
    voters = []

    def __init__(self):
        pass

    def addVoter(self,clf,method):
        self.voters.append(ClfMethod.ClfMethod(clf,method))

    def vote(self,datas):
        votes = [0,0]
        for voter in self.voters:
            data = datas[dataChoice[voter.method]]
            vote = voter.clf.predict([data])[0]
            votes[vote] = votes[vote] + 1
        if votes[0] > votes[1]:
            return 0
        if votes[1] > votes[0] or random.random() > 0.5:
            return 1
        return 0
