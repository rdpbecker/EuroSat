def zeroPad(string,length):
    copy = string
    while len(copy) < length:
        if len(copy)%2:
            copy = copy + " "
        else:
            copy = " " + copy
    return copy

class Table:
    pp = 0
    np = 0
    pn = 0
    nn = 0

    def __init__(self):
        pass

    def addObs(self,actual,classified):
        if not actual in [0,1] or not classified in [0,1]:
            print("Invalid argument: ", actual, ",", classified,")")
            return
        test = (actual,classified)
        if test == (0,0):
            self.nn = self.nn + 1
        elif test == (1,0):
            self.pn = self.pn + 1
        elif test == (0,1):
            self.np = self.np + 1
        elif test == (1,1):
            self.pp = self.pp + 1

    def specificity(self):
        return float(self.nn)/(self.np+self.nn)

    def sensitivity(self):
        return float(self.pp)/(self.pp+self.pn)

    def fScore(self):
        return 2*float(self.nn)/(self.np+self.pn+2*self.nn)

    def accuracy(self):
        return float(self.nn+self.pp)/(self.np+self.nn+self.pp+self.pn)

    def __str__(self):
        length = max([len(str(i)) for i in [self.pp,self.nn,self.np,self.pn]])
        string = "    |Pr +|Pr -|\n"
        string = string + "-"*(6+2*length)+"\n"
        string = string + "Tr +|" + zeroPad(str(self.pp),length) + "|" + zeroPad(str(self.pn),length) + "\n"
        string = string + "-"*(6+2*length)+"\n"
        string = string + "Tr -|" + zeroPad(str(self.np),length) + "|" + zeroPad(str(self.nn),length) + "\n\n"
        string = string + "Sensitivity: " + str(self.sensitivity()) + "\n"
        string = string + "Specificity: " + str(self.specificity()) + "\n"
        string = string + "F: " + str(self.fScore()) + "\n"
        string = string + "Accuracy: " + str(self.accuracy()) + "\n"
        return string

