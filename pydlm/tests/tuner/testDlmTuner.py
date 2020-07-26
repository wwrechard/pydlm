import unittest
from copy import deepcopy

import numpy as np

from pydlm.dlm import dlm
from pydlm.modeler.trends import trend
from pydlm.tuner.dlmTuner import modelTuner


class testModelTuner(unittest.TestCase):

    def setUp(self):
        self.mydlm = dlm(np.random.random(100)) + trend(2, discount=0.95)
        self.mytuner = modelTuner()

    def testFind_gradient(self):
        mydlm2 = deepcopy(self.mydlm)
        self.mydlm.fitForwardFilter()
        mydlm2.fitForwardFilter()
        mse0 = mydlm2._getMSE()
        mydlm2._setDiscounts(list(map(lambda x: x + self.mytuner.err,
                                      self.mydlm._getDiscounts())))
        mydlm2.fitForwardFilter()
        mse1 = mydlm2._getMSE()
        expect_gradient = (mse1 - mse0) / self.mytuner.err
        self.assertAlmostEqual(
            expect_gradient, self.mytuner.find_gradient(
                self.mydlm._getDiscounts(), self.mydlm))

    def testTune(self):
        tunedDLM = self.mytuner.tune(untunedDLM=self.mydlm, maxit=100)
        self.mydlm.fit()
        tunedDLM.fit()
        self.assertTrue(tunedDLM._getMSE() < self.mydlm._getMSE())
        self.assertTrue(
            max(tunedDLM._getDiscounts()) >= 1.0 - 2 * self.mytuner.err)


if __name__ == '__main__':
    unittest.main()
