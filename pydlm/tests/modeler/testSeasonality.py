import pydlm
import unittest

class testSeasonality(unittest.TestCase):

    def setUp(self):
        self.DEGREE = 7
        
    def testInitialization(self):
        newSeasonality = pydlm.modeler.seasonality.seasonality(self.DEGREE)
        newSeasonality.checkDimensions()

unittest.main()
        
