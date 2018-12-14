import unittest

from pydlm.modeler.trends import trend
from pydlm.modeler.seasonality import seasonality
from pydlm.modeler.autoReg import autoReg
from pydlm.func._dlmTune import _dlmTune


class test_dlmTune(unittest.TestCase):

    def setUp(self):
        self.data = [0] * 9 + [1] + [0] * 10
        self.data5 = range(100)
        self.dlm1 = _dlmTune(self.data)
        self.dlm2 = _dlmTune(self.data)
        self.dlm6 = _dlmTune(self.data5)
        self.dlm7 = _dlmTune([0, 1, None, 1, 0, 1, -1])
        self.dlm1.builder + trend(degree=0, discount=1, w=1.0)
        self.dlm2.builder + trend(degree=0, discount=1e-12, w=1.0)
        self.dlm6.builder + trend(degree=0, discount=0.9, w=1.0) + \
            seasonality(period=2, discount=0.8, w=1.0) + \
            autoReg(degree=3, discount=1.0)
        self.dlm7.builder + trend(degree=0, discount=1, w=1.0)
        self.dlm1._initialize()
        self.dlm2._initialize()
        self.dlm6._initialize()
        self.dlm7._initialize()
        self.dlm1.options.innovationType='whole'
        self.dlm2.options.innovationType='whole'
        self.dlm6.options.innovationType='whole'
        self.dlm7.options.innovationType='whole'

    def testComputeMSE(self):
        self.dlm1._forwardFilter(start=0, end=19, renew=False)
        self.dlm1.result.filteredSteps=(0, 19)
        mse1 = self.dlm1._getMSE()
        mse_expect = 0
        for i in range(20):
            mse_expect += (self.dlm1.result.predictedObs[i] -
                            self.data[i]) ** 2
        mse_expect /= 20
        self.assertAlmostEqual(mse1, mse_expect)

        self.dlm2._forwardFilter(start=0, end=19, renew=False)
        self.dlm2.result.filteredSteps=(0, 19)
        mse2 = self.dlm2._getMSE()
        mse_expect = 2.0/20
        self.assertAlmostEqual(mse2, mse_expect)

        # Test missing data
        self.dlm7._forwardFilter(start=0, end=6, renew=False)
        self.dlm7.result.filteredSteps=(0, 6)
        mse3 = self.dlm7._getMSE()
        mse_expect = 0
        for i in range(7):
            if self.dlm7.data[i] is not None:
                mse_expect += (self.dlm7.result.predictedObs[i] -
                               self.dlm7.data[i]) ** 2
        mse_expect /= 7
        self.assertAlmostEqual(mse3, mse_expect)

    def testGetDiscount(self):
        discounts = self.dlm6._getDiscounts()
        self.assertTrue(0.9 in discounts)
        self.assertTrue(0.8 in discounts)
        self.assertTrue(1.0 in discounts)
        
    def testSetDiscount(self):
        self.dlm6._setDiscounts([0.0, 0.1, 0.2], False)
        self.assertTrue(0.0 in self.dlm6.builder.discount)
        self.assertTrue(0.1 in self.dlm6.builder.discount)
        self.assertTrue(0.2 in self.dlm6.builder.discount)
        self.assertTrue(0.9 not in self.dlm6.builder.discount)
        self.assertTrue(0.8 not in self.dlm6.builder.discount)
        self.assertTrue(1.0 not in self.dlm6.builder.discount)

        self.assertAlmostEqual(self.dlm6.builder.staticComponents['trend'].discount,
                               0.9)

        self.dlm6._setDiscounts([0.0, 0.1, 0.2], True)
        self.assertTrue(self.dlm6.builder.staticComponents['trend'].discount in
                        [0.0, 0.1, 0.2])

if __name__ == '__main__':
    unittest.main()
