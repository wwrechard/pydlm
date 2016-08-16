import numpy as np
import unittest
from pydlm.modeler.builder import builder
from pydlm.modeler.trends import trend
from pydlm.modeler.seasonality import seasonality
from pydlm.modeler.dynamic import dynamic
from pydlm.modeler.matrixTools import matrixTools as mt

class testBuilder(unittest.TestCase):

    def setUp(self):
        self.features = np.random.rand(10, 2).tolist()
        self.trend = trend(degree = 3)
        self.seasonality = seasonality(period = 7)
        self.dynamic = dynamic(self.features)
        self.builder1 = builder()
        
    def testInitialization(self):
       
        self.assertEqual(len(self.builder1.dynamicComponents), 0)


    def testAddAndDelete(self):
        self.builder1 = self.builder1 + self.trend
        self.assertEqual(len(self.builder1.staticComponents), 1)
        self.assertEqual(len(self.builder1.dynamicComponents), 0)

        self.builder1 = self.builder1 + self.dynamic
        self.assertEqual(len(self.builder1.staticComponents), 1)
        self.assertEqual(len(self.builder1.dynamicComponents), 1)

        self.builder1 = self.builder1 + self.seasonality
        self.assertEqual(len(self.builder1.staticComponents), 2)
        self.assertEqual(len(self.builder1.dynamicComponents), 1)

        self.builder1.delete('seasonality')
        self.assertEqual(len(self.builder1.staticComponents), 1)
        self.assertEqual(len(self.builder1.dynamicComponents), 1)

        self.assertEqual(self.builder1.staticComponents['trend'], self.trend)

    
    def testInitialize(self):
        self.builder1 = self.builder1 + self.trend + self.dynamic
        
        self.builder1.initialize()
        self.assertAlmostEqual(np.sum(np.abs(self.builder1.model.evaluation - mt.matrixAddByCol(self.trend.evaluation, self.dynamic.evaluation))), 0.0)


    def testUpdate(self):
        self.builder1 = self.builder1 + self.trend + self.dynamic     
        self.builder1.initialize()

        self.builder1.updateEvaluation(2)
        self.assertAlmostEqual(np.sum(np.abs(self.builder1.model.evaluation - mt.matrixAddByCol(self.trend.evaluation, np.matrix([self.features[2]])))), 0.0)

unittest.main()
