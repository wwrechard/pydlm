import numpy as np
import unittest

from pydlm.modeler.trends import trend
from pydlm.func.dlm_base import dlm_base

class testDLM_base(unittest.TestCase):

    def setUp(self):
        self.data = [0] * 9 + [1] + [0] * 10
        self.dlm1 = dlm_base(self.data)
        self.dlm2 = dlm_base(self.data)
        self.dlm1.builder + trend(degree = 1, discount = 1)
        self.dlm2.builder + trend(degree = 1, discount = 1e-12)
        self.dlm1.__initialize__()
        self.dlm2.__initialize__()

    def testForwardFilter(self):
        self.dlm1.__forwardFilter__(start = 0, end = 19)
        self.assertAlmostEqual(np.sum(self.dlm1.result.filteredObs[0:9]), 0)
        self.assertAlmostEqual(self.dlm1.result.filteredObs[9], 1.0/11)
        self.assertAlmostEqual(self.dlm1.result.filteredObs[19], 1.0/21)

        self.dlm2.__forwardFilter__(start = 0, end = 19)
        self.assertAlmostEqual(np.sum(self.dlm2.result.filteredObs[0:9]), 0.0)
        self.assertAlmostEqual(self.dlm2.result.filteredObs[9], 1.0)
        self.assertAlmostEqual(self.dlm2.result.filteredObs[19], 0.0)        

    def testBackwardSmoother(self):
        self.dlm1.__forwardFilter__(start = 0, end = 19)
        self.dlm1.__backwardSmoother__(start = 19)
        self.assertAlmostEqual(self.dlm1.result.smoothedObs[0], 1.0/21)

        self.dlm2.__forwardFilter__(start = 0, end = 19)
        self.dlm2.__backwardSmoother__(start = 19)
        self.assertAlmostEqual(self.dlm2.result.smoothedObs[0], 0.0)
        
unittest.main()
