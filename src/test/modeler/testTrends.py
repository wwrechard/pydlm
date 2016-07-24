import sys
sys.path.append('/Users/samuel/Documents/Github/PyDLM/src/')

import pydlm

class testTrend:

    def __init__(self):
        self.DEGREE = 3
        
    def testTrendInitialization(self):
        newTrend = pydlm.modeler.trends.trend(self.DEGREE)
        newTrend.checkDimensions()
        
