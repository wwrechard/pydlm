import numpy as np
import unittest
from pydlm.modeler.builder import builder
from pydlm.modeler.trends import trend
from pydlm.modeler.seasonality import seasonality
from pydlm.modeler.dynamic import dynamic
from pydlm.modeler.autoReg import autoReg
from pydlm.modeler.matrixTools import matrixTools as mt


class testBuilder(unittest.TestCase):

    def setUp(self):
        self.data = np.random.rand(10).tolist()
        self.features = np.random.rand(10, 2).tolist()
        self.trend = trend(degree=2, w=1.0)
        self.seasonality = seasonality(period=7, w=1.0)
        self.dynamic = dynamic(self.features, w=1.0)
        self.autoReg = autoReg(degree=3,
                               w=1.0)
        self.builder1 = builder()

    def testInitialization(self):

        self.assertEqual(len(self.builder1.staticComponents), 0)
        self.assertEqual(len(self.builder1.dynamicComponents), 0)
        self.assertEqual(len(self.builder1.automaticComponents), 0)

    def testAddAndDelete(self):
        self.builder1 = self.builder1 + self.trend
        self.assertEqual(len(self.builder1.staticComponents), 1)
        self.assertEqual(len(self.builder1.dynamicComponents), 0)
        self.assertEqual(len(self.builder1.automaticComponents), 0)

        self.builder1 = self.builder1 + self.dynamic
        self.assertEqual(len(self.builder1.staticComponents), 1)
        self.assertEqual(len(self.builder1.dynamicComponents), 1)
        self.assertEqual(len(self.builder1.automaticComponents), 0)

        self.builder1 = self.builder1 + self.seasonality
        self.assertEqual(len(self.builder1.staticComponents), 2)
        self.assertEqual(len(self.builder1.dynamicComponents), 1)
        self.assertEqual(len(self.builder1.automaticComponents), 0)

        self.builder1.delete('seasonality')
        self.assertEqual(len(self.builder1.staticComponents), 1)
        self.assertEqual(len(self.builder1.dynamicComponents), 1)
        self.assertEqual(len(self.builder1.automaticComponents), 0)

        self.assertEqual(self.builder1.staticComponents['trend'], self.trend)

        self.builder1 = self.builder1 + self.autoReg
        self.assertEqual(len(self.builder1.automaticComponents), 1)

    def testInitialize(self):
        self.builder1 = self.builder1 + self.trend + self.dynamic \
                        + self.autoReg

        self.builder1.initialize(data=self.data)
        self.assertAlmostEqual(np.sum(
            np.abs(self.builder1.model.evaluation
                   - mt.matrixAddByCol(mt.matrixAddByCol(
                       self.trend.evaluation,
                       self.dynamic.evaluation),
                        self.autoReg.evaluation))), 0.0)

    def testInitializeEvaluatoin(self):
        self.builder1 = self.builder1 + self.trend + self.dynamic
        self.builder1.dynamicComponents['dynamic'].updateEvaluation(8)
        self.builder1.initialize(data=self.data)
        self.assertAlmostEqual(np.sum(
            np.abs(self.builder1.model.evaluation -
                   mt.matrixAddByCol(self.trend.evaluation,
                                     self.dynamic.evaluation))), 0.0)

    def testUpdate(self):
        self.builder1 = self.builder1 + self.trend + self.dynamic \
                        + self.autoReg

        self.builder1.initialize(data=self.data)
        self.builder1.updateEvaluation(2, self.data)
        self.assertAlmostEqual(np.sum(
            np.abs(self.builder1.model.evaluation
                   - mt.matrixAddByCol(mt.matrixAddByCol(
                       self.trend.evaluation,
                       np.matrix([self.features[2]])),
                                       np.matrix(self.autoReg.evaluation)))), 0.0)

if __name__ == '__main__':
    unittest.main()
