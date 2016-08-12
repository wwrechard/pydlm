import numpy as np
import unittest

from pydlm.modeler.trends import trend
from pydlm.modeler.seasonality import seasonality
from pydlm.dlm import dlm

class testDlm(unittest.TestCase):

    def setUp(self):
        self.data = [0] * 9 + [1] + [0] * 10
        self.dlm1 = dlm(self.data)
        self.dlm2 = dlm(self.data)
        self.dlm3 = dlm([-1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1])
        self.dlm1 + trend(degree = 1, discount = 1)
        self.dlm2 + trend(degree = 1, discount = 1e-12)
        self.dlm3 + seasonality(period = 2, discount = 1)

    def testFitForwardFilter(self):
        self.dlm1.fitForwardFilter(useRollingWindow = False)
        self.assertAlmostEqual(np.sum(self.dlm1.result.filteredObs[0:9]), 0)
        self.assertAlmostEqual(self.dlm1.result.filteredObs[9], 1.0/11)
        self.assertAlmostEqual(self.dlm1.result.filteredObs[19], 1.0/21)

        self.dlm2.fitForwardFilter(useRollingWindow = False)
        self.assertAlmostEqual(np.sum(self.dlm2.result.filteredObs[0:9]), 0.0)
        self.assertAlmostEqual(self.dlm2.result.filteredObs[9], 1.0)
        self.assertAlmostEqual(self.dlm2.result.filteredObs[19], 0.0)
        
    def testFitBackwardSmoother(self):
        self.dlm1.fitForwardFilter()
        self.dlm1.fitBackwardSmoother()
        self.assertAlmostEqual(self.dlm1.result.smoothedObs[0], 1.0/21)
        self.assertAlmostEqual(self.dlm1.result.smoothedObs[19], 1.0/21)

        self.dlm2.fitForwardFilter()
        self.dlm2.fitBackwardSmoother()
        self.assertAlmostEqual(self.dlm2.result.smoothedObs[0], 0.0)
        self.assertAlmostEqual(self.dlm2.result.smoothedObs[19], 0.0)
        self.assertAlmostEqual(self.dlm2.result.smoothedObs[9], 1.0)

unittest.main()
