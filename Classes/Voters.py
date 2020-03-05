import random

class Voters:
    voters = []

    def __init__(self):
        pass

    def addVoter(self,arr):
        new = {}
        for i in range(1,len(arr)):
            new[arr[i][0]] = arr[i][1]
        self.voters.append(new)

    def vote(self,ID):
        votes = [0,0]
        for voter in self.voters:
            vote = voter[ID]
            votes[vote] = votes[vote] + 1
        if votes[0] > votes[1]:
            return 0
        if votes[1] > votes[0] or random.random() > 0.5:
            return 1
        return 0
