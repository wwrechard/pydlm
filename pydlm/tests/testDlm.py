import numpy as np
import unittest

from pydlm.modeler.trends import trend
from pydlm.modeler.seasonality import seasonality
from pydlm.modeler.dynamic import dynamic
from pydlm.modeler.autoReg import autoReg
from pydlm.dlm import dlm

class testDlm(unittest.TestCase):

    def setUp(self):
        self.data = [0] * 9 + [1] + [0] * 10
        self.data5 = range(100)
        self.features = np.random.random((20, 2)).tolist()
        self.trend0 = trend(degree=0, discount=1.0, w=1.0)
        self.trend1 = trend(degree=0, discount=1.0)
        self.dlm1 = dlm(self.data)
        self.dlm2 = dlm(self.data)
        self.dlm3 = dlm([-1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1])
        self.dlm4 = dlm([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        self.dlm5 = dlm(self.data5)
        self.dlm6 = dlm(self.data5)
        self.dlm1 + trend(degree=0, discount=1, w=1.0)
        self.dlm2 + trend(degree=0, discount=1e-12, w=1.0)
        self.dlm3 + seasonality(period=2, discount=1, w=1.0)
        self.dlm4 + dynamic(features=[[0] for i in range(5)] +
                            [[1] for i in range(5)], discount=1, w=1.0)
        self.dlm5 + trend(degree=0, discount=1, w=1.0) + \
            autoReg(degree=1, discount=1, w=1.0)
        self.dlm6 + trend(degree=0, discount=1, w=1.0) + \
            autoReg(degree=2, discount=1, w=1.0)
        self.dlm1.evolveMode('dependent')
        self.dlm2.evolveMode('dependent')
        self.dlm3.evolveMode('dependent')
        self.dlm4.evolveMode('dependent')
        self.dlm5.evolveMode('dependent')
        self.dlm6.evolveMode('dependent')

    def testAdd(self):
        trend2 = trend(2, name='trend2')
        self.dlm1 = self.dlm1 + trend2
        self.assertEqual(self.dlm1.builder.staticComponents['trend2'], trend2)

        dynamic2 = dynamic(features=self.features, name='d2')
        self.dlm1 = self.dlm1 + dynamic2
        self.assertEqual(self.dlm1.builder.dynamicComponents['d2'], dynamic2)

        ar3 = autoReg(degree=3, name='ar3')
        self.dlm1 = self.dlm1 + ar3
        self.assertEqual(self.dlm1.builder.automaticComponents['ar3'], ar3)

    def testDelete(self):
        trend2 = trend(2, name='trend2')
        self.dlm1 = self.dlm1 + trend2
        self.dlm1.delete('trend2')
        self.assertEqual(len(self.dlm1.builder.staticComponents), 1)

    def testFitForwardFilter(self):
        self.dlm1.fitForwardFilter(useRollingWindow = False)
        self.assertEqual(self.dlm1.result.filteredSteps, [0, 19])
        self.assertAlmostEqual(np.sum(self.dlm1.result.filteredObs[0:9]), 0)
        self.assertAlmostEqual(self.dlm1.result.filteredObs[9][0, 0], 1.0/11)
        self.assertAlmostEqual(self.dlm1.result.filteredObs[19][0, 0], 1.0/21)

        self.dlm2.fitForwardFilter(useRollingWindow = False)
        self.assertAlmostEqual(np.sum(self.dlm2.result.filteredObs[0:9]), 0.0)
        self.assertAlmostEqual(self.dlm2.result.filteredObs[9][0, 0], 1.0)
        self.assertAlmostEqual(self.dlm2.result.filteredObs[19][0, 0], 0.0)

    def testFitBackwardSmoother(self):
        self.dlm1.fitForwardFilter()
        self.dlm1.fitBackwardSmoother()
        self.assertEqual(self.dlm1.result.smoothedSteps, [0, 19])
        self.assertAlmostEqual(self.dlm1.result.smoothedObs[0][0, 0], 1.0/21)
        self.assertAlmostEqual(self.dlm1.result.smoothedObs[19][0, 0], 1.0/21)

        self.dlm2.fitForwardFilter()
        self.dlm2.fitBackwardSmoother()
        self.assertAlmostEqual(self.dlm2.result.smoothedObs[0][0, 0], 0.0)
        self.assertAlmostEqual(self.dlm2.result.smoothedObs[19][0, 0], 0.0)
        self.assertAlmostEqual(self.dlm2.result.smoothedObs[9][0, 0], 1.0)

    def testAppend(self):
        dlm4 = dlm(self.data[0:11])
        dlm4 + self.trend0
        dlm4.evolveMode('dependent')
        dlm4.fitForwardFilter()
        self.assertEqual(dlm4.n, 11)

        dlm4.append(self.data[11 : 20])
        self.assertEqual(dlm4.n, 20)
        dlm4.fitForwardFilter()

        self.dlm1.fitForwardFilter()
        self.assertAlmostEqual(np.sum(np.array(dlm4.result.filteredObs) -
                                      np.array(self.dlm1.result.filteredObs)), 0.0)

    def testAppendDynamic(self):
        # we feed the data to dlm4 via two segments
        dlm4 = dlm(self.data[0:11])
        dlm4 + self.trend1 + dynamic(features = self.features[0:11],
                                     discount = 1)
        dlm4.fitForwardFilter()
        dlm4.append(self.data[11 : 20])
        dlm4.append(self.features[11 : 20], component = 'dynamic')
        dlm4.fitForwardFilter()

        # we feed the data to dlm5 all at once
        dlm5 = dlm(self.data)
        dlm5 + self.trend1 + dynamic(features = self.features,
                                     discount = 1)
        dlm5.fitForwardFilter()
        self.assertAlmostEqual(np.sum(np.array(dlm4.result.filteredObs) -
                                      np.array(dlm5.result.filteredObs)), 0.0)

    def testPopout(self):
        dlm4 = dlm(self.data)
        dlm4 + self.trend1 + dynamic(features = self.features, discount = 1)
        dlm4.fitForwardFilter()
        # the filtered step range should be (0, 19)
        self.assertEqual(dlm4.result.filteredSteps, [0, 19])

        # pop out the first date, the filtered range should be (0, -1)
        dlm4.popout(0)
        self.assertEqual(dlm4.result.filteredSteps, [0, -1])

        dlm4.fitForwardFilter()
        dlm5 = dlm(self.data[1 : 20])
        dlm5 + self.trend1 + dynamic(features = self.features[1 : 20],
                                     discount = 1)
        dlm5.fitForwardFilter()

        # The two chain should have the same filtered obs
        self.assertAlmostEqual(np.sum(np.array(dlm4.result.filteredObs) -
                                      np.array(dlm5.result.filteredObs)), 0.0)

    def testAlter(self):
        dlm4 = dlm(self.data)
        dlm4 + self.trend1 + dynamic(features = self.features, discount = 1)
        dlm4.fitForwardFilter()
        # the filtered step range should be (0, 19)
        self.assertEqual(dlm4.result.filteredSteps, [0, 19])

        dlm4.alter(date = 15, data = 1, component = 'main')
        self.assertEqual(dlm4.result.filteredSteps, [0, 14])
        dlm4.fitForwardFilter()

        newData = [0] * 9 + [1] + [0] * 10
        newData[15] = 1
        dlm5 = dlm(newData)
        dlm5 + self.trend1 + dynamic(features = self.features, discount = 1)
        dlm5.fitForwardFilter()

        # The two chain should have the same filtered obs
        self.assertAlmostEqual(np.sum(np.array(dlm4.result.filteredObs) - \
                                      np.array(dlm5.result.filteredObs)), 0.0)

        # test alter the feature
        dlm4.alter(date=0, data=[1,1], component='dynamic')
        self.assertAlmostEqual(dlm4.builder.dynamicComponents['dynamic'].features[0],
                               [1, 1])

    def testOneDayAheadPredictWithoutDynamic(self):
        self.dlm3.fitForwardFilter()
        (obs, var) = self.dlm3.predict(date=11)
        self.assertAlmostEqual(obs, -6.0/7)
        self.assertAlmostEqual(self.dlm3._predictModel.result.predictStatus,
                               [11, 12, [-6.0/7]])

        (obs, var) = self.dlm3.predict(date=2)
        self.assertAlmostEqual(obs, 3.0/5)
        # notice that the two latent states always sum up to 0
        self.assertAlmostEqual(self.dlm3._predictModel.result.predictStatus,
                               [2, 3, [3.0/5]])
        
    def testOneDayAheadPredictWithDynamic(self):
        self.dlm4.fitForwardFilter()
        featureDict = {'dynamic': 2.0}
        (obs, var) = self.dlm4.predict(date=9,
                                       featureDict=featureDict)
        self.assertAlmostEqual(obs, 5.0/6 * 2)

    def testContinuePredictWithoutDynamic(self):
        self.dlm3.fitForwardFilter()
        (obs, var) = self.dlm3.predict(date=11)
        self.assertAlmostEqual(self.dlm3._predictModel.result.predictStatus,
                               [11, 12, [-6.0/7]])
        (obs, var) = self.dlm3.continuePredict()
        self.assertAlmostEqual(self.dlm3._predictModel.result.predictStatus,
                               [11, 13, [-6.0/7, 6.0/7]])

    def testContinuePredictWithDynamic(self):
        self.dlm4.fitForwardFilter()
        featureDict = {'dynamic': [2.0]}
        (obs, var) = self.dlm4.predict(date=9,
                                       featureDict=featureDict)
        self.assertAlmostEqual(self.dlm4._predictModel.result.predictStatus,
                               [9, 10, [5.0/6 * 2]])

        featureDict = {'dynamic': [3.0]}
        (obs, var) = self.dlm4.continuePredict(featureDict=featureDict)
        self.assertAlmostEqual(self.dlm4._predictModel.result.predictStatus,
                               [9, 11, [5.0/6 * 2, 5.0/6 * 3]])

    def testPredictWithAutoReg(self):
        self.dlm5.fitForwardFilter()
        (obs, var) = self.dlm5.predict(date=99)
        self.assertAlmostEqual(obs[0, 0], 100.03682874)
        (obs, var) = self.dlm5.continuePredict()
        self.assertAlmostEqual(obs[0, 0], 101.07480945)

    def testPredictWithAutoReg2(self):
        self.dlm6.fitForwardFilter()
        (obs, var) = self.dlm6.predict(date=99)
        self.assertAlmostEqual(obs[0, 0], 100.02735)
        (obs, var) = self.dlm6.continuePredict()
        self.assertAlmostEqual(obs[0, 0], 101.06011996)
        (obs, var) = self.dlm6.continuePredict()
        self.assertAlmostEqual(obs[0, 0], 102.0946503)

    def testPredictNWithoutDynamic(self):
        self.dlm3.fitForwardFilter()
        (obs, var) = self.dlm3.predictN(N=2, date=11)
        self.assertAlmostEqual(self.dlm3._predictModel.result.predictStatus,
                               [11, 13, [-6.0/7, 6.0/7]])

    def testPredictNWithDynamic(self):
        self.dlm4.fitForwardFilter()
        featureDict = {'dynamic': [[2.0], [3.0]]}
        (obs, var) = self.dlm4.predictN(N=2, date=9,
                                        featureDict=featureDict)
        self.assertAlmostEqual(self.dlm4._predictModel.result.predictStatus,
                               [9, 11, [5.0/6 * 2, 5.0/6 * 3]])   

    def testPredictNWithAutoReg(self):
        self.dlm5.fitForwardFilter()
        (obs, var) = self.dlm5.predictN(N=2, date=99)
        self.assertAlmostEqual(obs[0], 100.03682874)
        self.assertAlmostEqual(obs[1], 101.07480945)

    def testPredictNWithDynamicMatrixInput(self):
        self.dlm4.fitForwardFilter()
        featureDict = {'dynamic': np.matrix([[2.0], [3.0]])}
        (obs, var) = self.dlm4.predictN(N=2, date=9,
                                        featureDict=featureDict)
        self.assertAlmostEqual(self.dlm4._predictModel.result.predictStatus,
                               [9, 11, [5.0/6 * 2, 5.0/6 * 3]])

    def testPredictionNotChangeModel(self):
        timeSeries = [1, 2, 1, 5, 3, 5, 4, 8, 1, 2]

        dlm1 = dlm(timeSeries) + trend(degree=2, discount=0.95)
        dlm1.fitForwardFilter()
        (obs1, var1) = dlm1.predictN(N=1, date=dlm1.n-1)

        dlm2 = dlm([]) + trend(degree=2, discount=0.95)
        for d in timeSeries:
            dlm2.append([d], component='main')
            dlm2.fitForwardFilter()
            (obs2, var2) = dlm2.predictN(N=1, date=dlm2.n-1)
        
        self.assertAlmostEqual(obs1, obs2)
        self.assertAlmostEqual(var1, var2)

    def testGetLatentState(self):
        # for forward filter
        self.dlm5.fitForwardFilter()
        filteredTrend = self.dlm5.getLatentState(
            filterType='forwardFilter', name='trend')
        diff = 0.0
        for i in range(len(filteredTrend)):
            diff += abs(filteredTrend[i][0] -
                        self.dlm5.result.filteredState[i][0, 0])
        self.assertAlmostEqual(diff, 0)

        # for backward smoother
        self.dlm5.fitBackwardSmoother()
        smoothedTrend = self.dlm5.getLatentState(
            filterType='backwardSmoother', name='trend')
        diff = 0.0
        for i in range(len(smoothedTrend)):
            diff += abs(smoothedTrend[i][0] -
                        self.dlm5.result.smoothedState[i][0, 0])
        self.assertAlmostEqual(diff, 0)

    def testGetLatentCov(self):
        # for forward filter
        self.dlm5.fitForwardFilter()
        filteredTrend = self.dlm5.getLatentCov(
            filterType='forwardFilter', name='trend')
        diff = 0.0
        for i in range(len(filteredTrend)):
            diff += abs(filteredTrend[i][0, 0] -
                        self.dlm5.result.filteredCov[i][0, 0])
        self.assertAlmostEqual(diff, 0)

        # for backward smoother
        self.dlm5.fitBackwardSmoother()
        smoothedTrend = self.dlm5.getLatentCov(
            filterType='backwardSmoother', name='trend')
        diff = 0.0
        for i in range(len(smoothedTrend)):
            diff += abs(smoothedTrend[i][0, 0] -
                        self.dlm5.result.smoothedCov[i][0, 0])
        self.assertAlmostEqual(diff, 0)

    def testGetMean(self):
        # for forward filter
        self.dlm5.fitForwardFilter()
        filteredTrend = self.dlm5.getMean(filterType='forwardFilter')
        self.assertEqual(len(filteredTrend), self.dlm5.n)
        diff = 0.0
        for i in range(len(filteredTrend)):
            diff += abs(filteredTrend[i] -
                        self.dlm5.result.filteredObs[i][0, 0])
        self.assertAlmostEqual(diff, 0)

        # for component with forward filter
        arTrend = self.dlm5.getMean(filterType='forwardFilter',
                                    name='ar2')
        trueAr = [item[1, 0] for item in self.dlm5.result.filteredState]
        comp = self.dlm5.builder.automaticComponents['ar2']
        for i in range(len(trueAr)):
            comp.updateEvaluation(i, self.data5)
            trueAr[i] = comp.evaluation * trueAr[i]

        diff = 0.0
        for i in range(len(arTrend)):
            diff += abs(arTrend[i] - trueAr[i])
        self.assertAlmostEqual(diff, 0)

        # for backward smoother
        self.dlm5.fitBackwardSmoother()
        filteredTrend = self.dlm5.getMean(filterType='backwardSmoother')
        diff = 0.0
        for i in range(len(filteredTrend)):
            diff += abs(filteredTrend[i] -
                        self.dlm5.result.smoothedObs[i][0, 0])
        self.assertAlmostEqual(diff, 0)

        # for component with backward smoother
        arTrend = self.dlm5.getMean(filterType='backwardSmoother',
                                    name='ar2')
        trueAr = [item[1, 0] for item in self.dlm5.result.smoothedState]
        comp = self.dlm5.builder.automaticComponents['ar2']
        for i in range(len(trueAr)):
            comp.updateEvaluation(i, self.data5)
            trueAr[i] = comp.evaluation * trueAr[i]

        diff = 0.0
        for i in range(len(arTrend)):
            diff += abs(arTrend[i] - trueAr[i])
        self.assertAlmostEqual(diff, 0)

    def testGetVar(self):
        # for forward filter
        self.dlm5.fitForwardFilter()
        filteredTrend = self.dlm5.getVar(filterType='forwardFilter')
        self.assertEqual(len(filteredTrend), self.dlm5.n)
        diff = 0.0
        for i in range(len(filteredTrend)):
            diff += abs(filteredTrend[i] -
                        self.dlm5.result.filteredObsVar[i][0, 0])
        self.assertAlmostEqual(diff, 0)

        # for component with forward filter
        arTrend = self.dlm5.getVar(filterType='forwardFilter',
                                    name='ar2')
        trueAr = [item[1, 1] for item in self.dlm5.result.filteredCov]
        comp = self.dlm5.builder.automaticComponents['ar2']
        for i in range(len(trueAr)):
            comp.updateEvaluation(i, self.data5)
            trueAr[i] = comp.evaluation * trueAr[i] * comp.evaluation.T

        diff = 0.0
        for i in range(len(arTrend)):
            diff += abs(arTrend[i] - trueAr[i])
        self.assertAlmostEqual(diff, 0)

        # for backward smoother
        self.dlm5.fitBackwardSmoother()
        filteredTrend = self.dlm5.getVar(filterType='backwardSmoother')
        diff = 0.0
        for i in range(len(filteredTrend)):
            diff += abs(filteredTrend[i] -
                        self.dlm5.result.smoothedObsVar[i][0, 0])
        self.assertAlmostEqual(diff, 0)

        # for component with backward smoother
        arTrend = self.dlm5.getVar(filterType='backwardSmoother',
                                    name='ar2')
        trueAr = [item[1, 1] for item in self.dlm5.result.smoothedCov]
        comp = self.dlm5.builder.automaticComponents['ar2']
        for i in range(len(trueAr)):
            comp.updateEvaluation(i, self.data5)
            trueAr[i] = comp.evaluation * trueAr[i] * comp.evaluation.T

        diff = 0.0
        for i in range(len(arTrend)):
            diff += abs(arTrend[i] - trueAr[i])
        self.assertAlmostEqual(diff, 0)

    def testGetMSE(self):
        self.dlm1.stableMode(False)
        self.dlm1.fitForwardFilter()
        mse1 = self.dlm1.getMSE()
        mse_expect = 0
        for i in range(20):
            mse_expect += (self.dlm1.result.predictedObs[i] -
                            self.data[i]) ** 2
        mse_expect /= 20
        self.assertAlmostEqual(mse1, mse_expect)

        self.dlm2.stableMode(False)
        self.dlm2.fitForwardFilter()
        self.dlm2.result.filteredSteps = (0, 19)
        mse2 = self.dlm2.getMSE()
        mse_expect = 2.0/20

        self.assertAlmostEqual(mse2, mse_expect)

    def testGetResidual(self):
        # for forward filter
        filter_type = 'forwardFilter'
        self.dlm5.fitForwardFilter()
        filteredTrend = self.dlm5.getMean(filterType=filter_type)
        filteredResidual = self.dlm5.getResidual(filterType=filter_type)
        self.assertEqual(len(filteredResidual), self.dlm5.n)
        diff = 0.0
        for i in range(len(filteredTrend)):
            diff += abs(- filteredTrend[i] - filteredResidual[i] +
                        self.dlm5.data[i])
        self.assertAlmostEqual(diff, 0)

        # for backward smoother
        filter_type = 'backwardSmoother'
        self.dlm5.fitBackwardSmoother()
        filteredTrend = self.dlm5.getMean(filterType=filter_type)
        filteredResidual = self.dlm5.getResidual(filterType=filter_type)
        diff = 0.0
        for i in range(len(filteredTrend)):
            diff += abs(- filteredTrend[i] - filteredResidual[i] +
                        self.dlm5.data[i])
        self.assertAlmostEqual(diff, 0) 

    def testTune(self):
        # just make sure the tune can run
        self.dlm5.fit()
        self.dlm5.tune(maxit=10)

if __name__ == '__main__':
    unittest.main()
