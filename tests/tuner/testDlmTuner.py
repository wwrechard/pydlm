import unittest
import numpy as np

from copy import deepcopy
from pydlm.tuner.dlmTuner import modelTuner
from pydlm.modeler.trends import trend
from pydlm.dlm import dlm


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


if __name__ == '__main__':
    unittest.main()
