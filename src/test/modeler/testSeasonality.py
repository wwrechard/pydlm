import sys
sys.path.append('/Users/samuel/Documents/Github/PyDLM/src/')

import pydlm

class testSeasonality:

    def __init__(self):
        self.DEGREE = 7
        
    def testInitialization(self):
        newSeasonality = pydlm.modeler.seasonality.seasonality(self.DEGREE)
        newSeasonality.checkDimensions()
        return newSeasonality
        
