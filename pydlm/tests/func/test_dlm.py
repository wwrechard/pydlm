import numpy as np
import unittest

from pydlm.modeler.trends import trend
from pydlm.modeler.seasonality import seasonality
from pydlm.modeler.dynamic import dynamic
from pydlm.modeler.autoReg import autoReg
from pydlm.func._dlm import _dlm


class test_dlm(unittest.TestCase):

    def setUp(self):
        self.data = [0] * 9 + [1] + [0] * 10
        self.dlm1 = _dlm(self.data)
        self.dlm2 = _dlm(self.data)
        self.dlm3 = _dlm([-1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1])
        self.dlm4 = _dlm([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        self.dlm5 = _dlm(range(100))
        self.dlm6 = _dlm(range(100))
        self.dlm7 = _dlm([0, 1, None, 1, 0, 1, -1])
        self.dlm1.builder + trend(degree=0, discount=1, w=1.0)
        self.dlm2.builder + trend(degree=0, discount=1e-12, w=1.0)
        self.dlm3.builder + seasonality(period=2, discount=1, w=1.0)
        self.dlm4.builder + dynamic(features=[[0] for i in range(5)] +
                                    [[1] for i in range(5)], discount=1,
                                    w=1.0)
        self.dlm5.builder + trend(degree=0, discount=1, w=1.0) + \
            autoReg(degree=1, data=range(100), discount=1, w=1.0)
        self.dlm6.builder + trend(degree=0, discount=0.9, w=1.0) + \
            seasonality(period=2, discount=0.8, w=1.0) + \
            autoReg(degree=3, data=range(100), discount=1.0)
        self.dlm7.builder + trend(degree=0, discount=1, w=1.0)
        self.dlm1._initialize()
        self.dlm2._initialize()
        self.dlm3._initialize()
        self.dlm4._initialize()
        self.dlm5._initialize()
        self.dlm6._initialize()
        self.dlm7._initialize()
        self.dlm1.options.innovationType='whole'
        self.dlm2.options.innovationType='whole'
        self.dlm3.options.innovationType='whole'
        self.dlm4.options.innovationType='whole'
        self.dlm5.options.innovationType='whole'
        self.dlm6.options.innovationType='whole'
        self.dlm7.options.innovationType='whole'

    def testForwardFilter(self):
        self.dlm1._forwardFilter(start=0, end=19, renew=False)
        self.assertAlmostEqual(np.sum(self.dlm1.result.filteredObs[0:9]), 0)
        self.assertAlmostEqual(self.dlm1.result.filteredObs[9][0, 0], 1.0/11)
        self.assertAlmostEqual(self.dlm1.result.filteredObs[19][0, 0], 1.0/21)

        self.dlm2._forwardFilter(start=0, end=19)
        self.assertAlmostEqual(np.sum(self.dlm2.result.filteredObs[0:9]), 0.0)
        self.assertAlmostEqual(self.dlm2.result.filteredObs[9][0, 0], 1.0)
        self.assertAlmostEqual(self.dlm2.result.filteredObs[19][0, 0], 0.0)

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
        self.assertAlmostEqual(self.dlm1.result.smoothedObs[0][0, 0], 1.0/21)
        self.assertAlmostEqual(self.dlm1.result.smoothedObs[19][0, 0], 1.0/21)

        self.dlm2._forwardFilter(start = 0, end = 19)
        self.dlm2.result.filteredSteps = (0, 19)
        self.dlm2._backwardSmoother(start = 19)
        self.assertAlmostEqual(self.dlm2.result.smoothedObs[0][0, 0], 0.0)
        self.assertAlmostEqual(self.dlm2.result.smoothedObs[19][0, 0], 0.0)
        self.assertAlmostEqual(self.dlm2.result.smoothedObs[9][0, 0], 1.0)

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

    def testGetLatentState(self):
        # for forward filter
        self.dlm5._forwardFilter(start=0, end=99, renew=False)
        self.dlm5.result.filteredSteps = [0, 99]
        filteredTrend = self.dlm5._getLatentState(
            filterType='forwardFilter', name='trend', start=0, end=99)
        diff = 0.0
        for i in range(len(filteredTrend)):
            diff += abs(filteredTrend[i][0] -
                        self.dlm5.result.filteredState[i][0, 0])
        self.assertAlmostEqual(diff, 0)

        # for prediction
        predictedTrend = self.dlm5._getLatentState(
            filterType='predict', name='trend', start=0, end=99)
        diff = 0.0
        for i in range(len(predictedTrend)):
            diff += abs(predictedTrend[i][0] -
                        self.dlm5.result.predictedState[i][0, 0])
        self.assertAlmostEqual(diff, 0)

        # for backward smoother
        self.dlm5._backwardSmoother(start=99)
        self.dlm5.result.smoothedSteps = [0, 99]
        smoothedTrend = self.dlm5._getLatentState(
            filterType='backwardSmoother', name='trend', start=0, end=99)
        diff = 0.0
        for i in range(len(smoothedTrend)):
            diff += abs(smoothedTrend[i][0] -
                        self.dlm5.result.smoothedState[i][0, 0])
        self.assertAlmostEqual(diff, 0)

    def testGetLatentCov(self):
        # for forward filter
        self.dlm5._forwardFilter(start=0, end=99, renew=False)
        self.dlm5.result.filteredSteps = [0, 99]
        filteredTrend = self.dlm5._getLatentCov(
            filterType='forwardFilter', name='trend', start=0, end=99)
        diff = 0.0
        for i in range(len(filteredTrend)):
            diff += abs(filteredTrend[i][0, 0] -
                        self.dlm5.result.filteredCov[i][0, 0])
        self.assertAlmostEqual(diff, 0)

        # for prediction
        predictedTrend = self.dlm5._getLatentCov(
            filterType='predict', name='trend', start=0, end=99)
        diff = 0.0
        for i in range(len(predictedTrend)):
            diff += abs(predictedTrend[i][0, 0] -
                        self.dlm5.result.predictedCov[i][0, 0])
        self.assertAlmostEqual(diff, 0)

        # for backward smoother
        self.dlm5._backwardSmoother(start=99)
        self.dlm5.result.smoothedSteps = [0, 99]
        smoothedTrend = self.dlm5._getLatentCov(
            filterType='backwardSmoother', name='trend', start=0, end=99)
        diff = 0.0
        for i in range(len(smoothedTrend)):
            diff += abs(smoothedTrend[i][0, 0] -
                        self.dlm5.result.smoothedCov[i][0, 0])
        self.assertAlmostEqual(diff, 0)

    def testComponentMean(self):
        self.dlm5._forwardFilter(start=0, end=99, renew=False)
        self.dlm5.result.filteredSteps = [0, 99]
        # for component with forward filter
        arTrend = self.dlm5._getComponentMean(filterType='forwardFilter',
                                              name='ar2', start=0, end=99)
        trueAr = [item[1, 0] for item in self.dlm5.result.filteredState]
        comp = self.dlm5.builder.automaticComponents['ar2']
        for i in range(len(trueAr)):
            comp.updateEvaluation(i)
            trueAr[i] = comp.evaluation * trueAr[i]

        diff = 0.0
        for i in range(len(arTrend)):
            diff += abs(arTrend[i] - trueAr[i])
        self.assertAlmostEqual(diff, 0)

        # for component with backward smoother
        self.dlm5._backwardSmoother(start=99)
        self.dlm5.result.smoothedSteps = [0, 99]
        arTrend = self.dlm5._getComponentMean(filterType='backwardSmoother',
                                    name='ar2', start=0, end=99)
        trueAr = [item[1, 0] for item in self.dlm5.result.smoothedState]
        comp = self.dlm5.builder.automaticComponents['ar2']
        for i in range(len(trueAr)):
            comp.updateEvaluation(i)
            trueAr[i] = comp.evaluation * trueAr[i]

        diff = 0.0
        for i in range(len(arTrend)):
            diff += abs(arTrend[i] - trueAr[i])
        self.assertAlmostEqual(diff, 0)

    def testComponentVar(self):
        self.dlm5._forwardFilter(start=0, end=99, renew=False)
        self.dlm5.result.filteredSteps = [0, 99]
        # for component with forward filter
        arTrend = self.dlm5._getComponentVar(filterType='forwardFilter',
                                             name='ar2', start=0, end=99)
        trueAr = [item[1, 1] for item in self.dlm5.result.filteredCov]
        comp = self.dlm5.builder.automaticComponents['ar2']
        for i in range(len(trueAr)):
            comp.updateEvaluation(i)
            trueAr[i] = comp.evaluation * trueAr[i] * comp.evaluation.T

        diff = 0.0
        for i in range(len(arTrend)):
            diff += abs(arTrend[i] - trueAr[i])
        self.assertAlmostEqual(diff, 0)

        # for component with backward smoother
        self.dlm5._backwardSmoother(start=99)
        self.dlm5.result.smoothedSteps = [0, 99]
        arTrend = self.dlm5._getComponentVar(filterType='backwardSmoother',
                                             name='ar2', start=0, end=99)
        trueAr = [item[1, 1] for item in self.dlm5.result.smoothedCov]
        comp = self.dlm5.builder.automaticComponents['ar2']
        for i in range(len(trueAr)):
            comp.updateEvaluation(i)
            trueAr[i] = comp.evaluation * trueAr[i] * comp.evaluation.T

        diff = 0.0
        for i in range(len(arTrend)):
            diff += abs(arTrend[i] - trueAr[i])
        self.assertAlmostEqual(diff, 0)

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
