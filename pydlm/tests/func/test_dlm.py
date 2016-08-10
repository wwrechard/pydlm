import numpy as np
import unittest

from pydlm.modeler.trends import trend
from pydlm.func._dlm import _dlm

class test_dlm(unittest.TestCase):

    def setUp(self):
        self.data = [0] * 9 + [1] + [0] * 10
        self.dlm1 = _dlm(self.data)
        self.dlm2 = _dlm(self.data)
        self.dlm1.builder + trend(degree = 1, discount = 1)
        self.dlm2.builder + trend(degree = 1, discount = 1e-12)
        self.dlm1._initialize()
        self.dlm2._initialize()

    def testForwardFilter(self):
        self.dlm1._forwardFilter(start = 0, end = 19)
        self.assertAlmostEqual(np.sum(self.dlm1.result.filteredObs[0:9]), 0)
        self.assertAlmostEqual(self.dlm1.result.filteredObs[9], 1.0/11)
        self.assertAlmostEqual(self.dlm1.result.filteredObs[19], 1.0/21)

        self.dlm2._forwardFilter(start = 0, end = 19)
        self.assertAlmostEqual(np.sum(self.dlm2.result.filteredObs[0:9]), 0.0)
        self.assertAlmostEqual(self.dlm2.result.filteredObs[9], 1.0)
        self.assertAlmostEqual(self.dlm2.result.filteredObs[19], 0.0)        

    def testBackwardSmoother(self):
        self.dlm1._forwardFilter(start = 0, end = 19)
        self.dlm1._backwardSmoother(start = 19)
        self.assertAlmostEqual(self.dlm1.result.smoothedObs[0], 1.0/21)

        self.dlm2._forwardFilter(start = 0, end = 19)
        self.dlm2._backwardSmoother(start = 19)
        self.assertAlmostEqual(self.dlm2.result.smoothedObs[0], 0.0)
        
unittest.main()
