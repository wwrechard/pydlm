import pydlm
import unittest

from pydlm.modeler.seasonality import seasonality

class testSeasonality(unittest.TestCase):


    def testInitialization(self):
        newSeasonality = seasonality(period=7)
        newSeasonality.checkDimensions()


    def testFreeForm(self):
        seasonality2 = seasonality(period=2, discount=1, w=1.0)
        self.assertEqual(seasonality2.meanPrior.tolist(), [[0], [0]])
        self.assertEqual(seasonality2.covPrior.tolist(), [[0.5, -0.5], [-0.5, 0.5]])


if __name__ == '__main__':
    unittest.main()
