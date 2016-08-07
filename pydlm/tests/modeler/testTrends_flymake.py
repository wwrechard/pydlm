import pydlm

class testTrend:

    def __init__(self):
        self.DEGREE = 3
        
    def testInitialization(self):
        newTrend = pydlm.modeler.trends.trend(self.DEGREE)
        newTrend.checkDimensions()
