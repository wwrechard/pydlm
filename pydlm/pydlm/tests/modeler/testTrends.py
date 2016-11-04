import pydlm
import unittest

class testTrend(unittest.TestCase):

    def setUp(self):
        self.DEGREE = 3
        
    def testInitialization(self):
        newTrend = pydlm.modeler.trends.trend(self.DEGREE)
        newTrend.checkDimensions()

unittest.main()
