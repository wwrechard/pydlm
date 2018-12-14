import unittest

from pydlm.modeler.trends import trend
from pydlm.modeler.seasonality import seasonality
from pydlm.modeler.dynamic import dynamic
from pydlm.modeler.autoReg import autoReg
from pydlm.func._dlmPredict import _dlmPredict


class test_dlmPredict(unittest.TestCase):

    def setUp(self):
        self.data = [0] * 9 + [1] + [0] * 10
        self.data5 = range(100)
        self.dlm3 = _dlmPredict([-1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1])
        self.dlm4 = _dlmPredict([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        self.dlm5 = _dlmPredict(self.data5)

        self.dlm3.builder + seasonality(period=2, discount=1, w=1.0)
        self.dlm4.builder + dynamic(features=[[0] for i in range(5)] +
                                    [[1] for i in range(5)], discount=1,
                                    w=1.0)
        self.dlm5.builder + trend(degree=0, discount=1, w=1.0) + \
            autoReg(degree=1, discount=1, w=1.0)

        self.dlm3._initialize()
        self.dlm4._initialize()
        self.dlm5._initialize()

        self.dlm3.options.innovationType='whole'
        self.dlm4.options.innovationType='whole'
        self.dlm5.options.innovationType='whole'

    def testOneDayAheadPredictWithoutDynamic(self):
        self.dlm3._forwardFilter(start=0, end=11, renew=False)
        self.dlm3.result.filteredSteps = (0, 11)
        (obs, var) = self.dlm3._oneDayAheadPredict(date=11)
        self.assertAlmostEqual(obs, -6.0/7)
        self.assertAlmostEqual(self.dlm3.result.predictStatus,
                               [11, 12, [-6.0/7]])

        (obs, var) = self.dlm3._oneDayAheadPredict(date=2)
        self.assertAlmostEqual(obs, 3.0/5)
        # notice that the two latent states always sum up to 0
        self.assertAlmostEqual(self.dlm3.result.predictStatus,
                               [2, 3, [3.0/5]])

    def testOneDayAheadPredictWithDynamic(self):
        self.dlm4._forwardFilter(start=0, end=9, renew=False)
        self.dlm4.result.filteredSteps = (0, 9)
        featureDict = {'dynamic': 2.0}
        (obs, var) = self.dlm4._oneDayAheadPredict(date=9,
                                                   featureDict=featureDict)
        self.assertAlmostEqual(obs, 5.0/6 * 2)

    def testContinuePredictWithoutDynamic(self):
        self.dlm3._forwardFilter(start=0, end=11, renew=False)
        self.dlm3.result.filteredSteps = (0, 11)
        (obs, var) = self.dlm3._oneDayAheadPredict(date=11)
        self.assertAlmostEqual(self.dlm3.result.predictStatus,
                               [11, 12, [-6.0/7]])
        (obs, var) = self.dlm3._continuePredict()
        self.assertAlmostEqual(self.dlm3.result.predictStatus,
                               [11, 13, [-6.0/7, 6.0/7]])

    def testContinuePredictWithDynamic(self):
        self.dlm4._forwardFilter(start=0, end=9, renew=False)
        self.dlm4.result.filteredSteps = (0, 9)
        featureDict = {'dynamic': 2.0}
        (obs, var) = self.dlm4._oneDayAheadPredict(date=9,
                                                   featureDict=featureDict)
        self.assertAlmostEqual(self.dlm4.result.predictStatus,
                               [9, 10, [5.0/6 * 2]])

        featureDict = {'dynamic': 3.0}
        (obs, var) = self.dlm4._continuePredict(featureDict=featureDict)
        self.assertAlmostEqual(self.dlm4.result.predictStatus,
                               [9, 11, [5.0/6 * 2, 5.0/6 * 3]])

    def testPredictWithAutoReg(self):
        self.dlm5._forwardFilter(start=0, end=99, renew=False)
        self.dlm5.result.filteredSteps = [0, 99]
        (obs, var) = self.dlm5._oneDayAheadPredict(date=99)
        self.assertAlmostEqual(obs[0, 0], 100.03682874)
        (obs, var) = self.dlm5._continuePredict()
        self.assertAlmostEqual(obs[0, 0], 101.07480945)

if __name__ == '__main__':
    unittest.main()
