###############################################################
## A class to hold the enumeration of the string values in a 
## column (as a dictionary) and the integer values in the 
## same column (as a list)
###############################################################

class CompleteEnum:
    enum = {}
    ints = []

    def __init__(self,enum,ints):
        self.enum = enum
        self.ints = ints


