import numpy as np
import unittest

from pydlm.modeler.trends import trend
from pydlm.modeler.seasonality import seasonality
from pydlm.func._dlm import _dlm

class test_dlm(unittest.TestCase):

    def setUp(self):
        self.data = [0] * 9 + [1] + [0] * 10
        self.dlm1 = _dlm(self.data)
        self.dlm2 = _dlm(self.data)
        self.dlm3 = _dlm([-1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1])
        self.dlm1.builder + trend(degree = 1, discount = 1)
        self.dlm2.builder + trend(degree = 1, discount = 1e-12)
        self.dlm3.builder + seasonality(period = 2, discount = 1)
        self.dlm1._initialize()
        self.dlm2._initialize()
        self.dlm3._initialize()

    def testForwardFilter(self):
        self.dlm1._forwardFilter(start = 0, end = 19, renew = False)
        self.assertAlmostEqual(np.sum(self.dlm1.result.filteredObs[0:9]), 0)
        self.assertAlmostEqual(self.dlm1.result.filteredObs[9], 1.0/11)
        self.assertAlmostEqual(self.dlm1.result.filteredObs[19], 1.0/21)

        self.dlm2._forwardFilter(start = 0, end = 19)
        self.assertAlmostEqual(np.sum(self.dlm2.result.filteredObs[0:9]), 0.0)
        self.assertAlmostEqual(self.dlm2.result.filteredObs[9], 1.0)
        self.assertAlmostEqual(self.dlm2.result.filteredObs[19], 0.0)

    def testResetModelStatus(self):
        self.dlm1._forwardFilter(start = 0, end = 19, renew = False)
        self.dlm1.result.filteredSteps = (0, 19)
        self.assertAlmostEqual(self.dlm1.builder.model.obs, \
                               self.dlm1.result.filteredObs[19])
        
        self.dlm1._resetModelStatus()
        self.assertAlmostEqual(np.sum(self.dlm1.builder.model.state \
                                      - self.dlm1.builder.statePrior), 0.0)        
    def testSetModelStatus(self):
        self.dlm1._forwardFilter(start = 0, end = 19, renew = False)
        self.dlm1.result.filteredSteps = (0, 19)
        self.assertAlmostEqual(self.dlm1.builder.model.obs, \
                               self.dlm1.result.filteredObs[19])
        self.dlm1._setModelStatus(date = 12)
        self.assertAlmostEqual(self.dlm1.builder.model.obs, \
                               self.dlm1.result.filteredObs[12])
        
    def testForwaredFilterConsectiveness(self):
        self.dlm1._forwardFilter(start = 0, end = 19, renew = False)
        filtered1 = self.dlm1.result.filteredObs

        self.dlm1._initialize()
        
        self.dlm1._forwardFilter(start = 0, end = 13)
        self.dlm1.result.filteredSteps = (0, 13)
        self.dlm1._forwardFilter(start = 13, end = 19)
        filtered2 = self.dlm1.result.filteredObs

        self.assertAlmostEqual(np.sum(np.array(filtered1) - np.array(filtered2)), 0.0)

    def testBackwardSmoother(self):
        self.dlm1._forwardFilter(start = 0, end = 19, renew = False)
        self.dlm1.result.filteredSteps = (0, 19)
        self.dlm1._backwardSmoother(start = 19)
        self.assertAlmostEqual(self.dlm1.result.smoothedObs[0], 1.0/21)
        self.assertAlmostEqual(self.dlm1.result.smoothedObs[19], 1.0/21)

        self.dlm2._forwardFilter(start = 0, end = 19)
        self.dlm2.result.filteredSteps = (0, 19)
        self.dlm2._backwardSmoother(start = 19)
        self.assertAlmostEqual(self.dlm2.result.smoothedObs[0], 0.0)
        self.assertAlmostEqual(self.dlm2.result.smoothedObs[19], 0.0)
        self.assertAlmostEqual(self.dlm2.result.smoothedObs[9], 1.0)

    def testPredict(self):
        self.dlm3._forwardFilter(start = 0, end = 11, renew = False)
        self.dlm3.result.filteredSteps = (0, 11)
        (obs, var) = self.dlm3._predict(days = 1)
        self.assertAlmostEqual(obs[0], -6.0/7)

unittest.main()
