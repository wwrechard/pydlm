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
        self.features = np.random.random((20, 2)).tolist()
        self.dlm1 = dlm(self.data)
        self.dlm2 = dlm(self.data)
        self.dlm3 = dlm([-1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1])
        self.dlm4 = dlm([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        self.dlm5 = dlm(range(100))
        self.dlm1 + trend(degree = 1, discount = 1)
        self.dlm2 + trend(degree = 1, discount = 1e-12)
        self.dlm3 + seasonality(period = 2, discount = 1)
        self.dlm4 + dynamic(features=[[0] for i in range(5)] +
                            [[1] for i in range(5)], discount=1)
        self.dlm5 + trend(degree=1, discount=1) + \
            autoReg(degree=1, data=range(100), discount=1)

    def testAdd(self):
        trend2 = trend(2, name='trend2')
        self.dlm1 = self.dlm1 + trend2
        self.assertEqual(self.dlm1.builder.staticComponents['trend2'], trend2)

        dynamic2 = dynamic(features=self.features, name='d2')
        self.dlm1 = self.dlm1 + dynamic2
        self.assertEqual(self.dlm1.builder.dynamicComponents['d2'], dynamic2)

        ar3 = autoReg(degree=3,data=self.data, name='ar3')
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
        self.assertAlmostEqual(self.dlm1.result.filteredObs[9], 1.0/11)
        self.assertAlmostEqual(self.dlm1.result.filteredObs[19], 1.0/21)

        self.dlm2.fitForwardFilter(useRollingWindow = False)
        self.assertAlmostEqual(np.sum(self.dlm2.result.filteredObs[0:9]), 0.0)
        self.assertAlmostEqual(self.dlm2.result.filteredObs[9], 1.0)
        self.assertAlmostEqual(self.dlm2.result.filteredObs[19], 0.0)

    def testFitBackwardSmoother(self):
        self.dlm1.fitForwardFilter()
        self.dlm1.fitBackwardSmoother()
        self.assertEqual(self.dlm1.result.smoothedSteps, [0, 19])
        self.assertAlmostEqual(self.dlm1.result.smoothedObs[0], 1.0/21)
        self.assertAlmostEqual(self.dlm1.result.smoothedObs[19], 1.0/21)

        self.dlm2.fitForwardFilter()
        self.dlm2.fitBackwardSmoother()
        self.assertAlmostEqual(self.dlm2.result.smoothedObs[0], 0.0)
        self.assertAlmostEqual(self.dlm2.result.smoothedObs[19], 0.0)
        self.assertAlmostEqual(self.dlm2.result.smoothedObs[9], 1.0)

    def testAppend(self):
        dlm4 = dlm(self.data[0:11])
        dlm4 + trend(degree = 1, discount = 1)
        dlm4.fitForwardFilter()
        self.assertEqual(dlm4.n, 11)

        dlm4.append(self.data[11 : 20])
        self.assertEqual(dlm4.n, 20)
        dlm4.fitForwardFilter()

        self.dlm1.fitForwardFilter()
        self.assertAlmostEqual(np.sum(np.array(dlm4.result.filteredObs) - \
                                      np.array(self.dlm1.result.filteredObs)), 0.0)

    def testAppendDynamic(self):
        # we feed the data to dlm4 via two segments
        dlm4 = dlm(self.data[0:11])
        dlm4 + trend(degree = 1, discount = 1) + dynamic(features = self.features[0:11], \
                                                         discount = 1)
        dlm4.fitForwardFilter()
        dlm4.append(self.data[11 : 20])
        dlm4.append(self.features[11 : 20], component = 'dynamic')
        dlm4.fitForwardFilter()

        # we feed the data to dlm5 all at once
        dlm5 = dlm(self.data)
        dlm5 + trend(degree = 1, discount = 1) + dynamic(features = self.features, \
                                                         discount = 1)
        dlm5.fitForwardFilter()
        self.assertAlmostEqual(np.sum(np.array(dlm4.result.filteredObs) - \
                                      np.array(dlm5.result.filteredObs)), 0.0)

    def testAppendAutomatic(self):
        # we feed the data to dlm4 via two segments
        dlm4 = dlm(self.data[0:11])
        dlm4 + trend(degree = 1, discount = 1) + autoReg(degree = 3,
                                                         data = self.data[0:11],
                                                         discount = 1)
        dlm4.fitForwardFilter()
        dlm4.append(self.data[11 : 20])
        dlm4.fitForwardFilter()

        # we feed the data to dlm5 all at once
        dlm5 = dlm(self.data)
        dlm5 + trend(degree = 1, discount = 1) + autoReg(degree = 3,
                                                         data = self.data,
                                                         discount = 1)
        dlm5.fitForwardFilter()
        self.assertAlmostEqual(np.sum(np.array(dlm4.result.filteredObs) - \
                                      np.array(dlm5.result.filteredObs)), 0.0)

    def testPopout(self):
        dlm4 = dlm(self.data)
        dlm4 + trend(degree = 1, discount = 1) + dynamic(features = \
                                                         self.features, \
                                                         discount = 1)
        dlm4.fitForwardFilter()
        # the filtered step range should be (0, 19)
        self.assertEqual(dlm4.result.filteredSteps, [0, 19])

        # pop out the first date, the filtered range should be (0, -1)
        dlm4.popout(0)
        self.assertEqual(dlm4.result.filteredSteps, [0, -1])

        dlm4.fitForwardFilter()
        dlm5 = dlm(self.data[1 : 20])
        dlm5 + trend(degree = 1, discount = 1) + dynamic(features = \
                                                         self.features[1 : 20], \
                                                         discount = 1)
        dlm5.fitForwardFilter()

        # The two chain should have the same filtered obs
        self.assertAlmostEqual(np.sum(np.array(dlm4.result.filteredObs) - \
                                      np.array(dlm5.result.filteredObs)), 0.0)

    def testAlter(self):
        dlm4 = dlm(self.data)
        dlm4 + trend(degree = 1, discount = 1) + dynamic(features = \
                                                         self.features, \
                                                         discount = 1)
        dlm4.fitForwardFilter()
        # the filtered step range should be (0, 19)
        self.assertEqual(dlm4.result.filteredSteps, [0, 19])

        # pop out the first date, the filtered range should be (0, -1)
        dlm4.alter(date = 15, data = 1, component = 'main')
        self.assertEqual(dlm4.result.filteredSteps, [0, 14])
        dlm4.fitForwardFilter()

        newData = [0] * 9 + [1] + [0] * 10
        newData[15] = 1
        dlm5 = dlm(newData)
        dlm5 + trend(degree = 1, discount = 1) + dynamic(features = \
                                                         self.features, \
                                                         discount = 1)
        dlm5.fitForwardFilter()

        # The two chain should have the same filtered obs
        self.assertAlmostEqual(np.sum(np.array(dlm4.result.filteredObs) - \
                                      np.array(dlm5.result.filteredObs)), 0.0)

    def testOneDayAheadPredictWithoutDynamic(self):
        self.dlm3.fitForwardFilter()
        (obs, var) = self.dlm3.predict(date=11)
        self.assertAlmostEqual(obs, -6.0/7)
        self.assertAlmostEqual(self.dlm3.result.predictStatus,
                               [11, 12, [-6.0/7]])

        (obs, var) = self.dlm3.predict(date=2)
        self.assertAlmostEqual(obs, 3.0/5)
        # notice that the two latent states always sum up to 0
        self.assertAlmostEqual(self.dlm3.result.predictStatus,
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
        self.assertAlmostEqual(self.dlm3.result.predictStatus,
                               [11, 12, [-6.0/7]])
        (obs, var) = self.dlm3.continuePredict()
        self.assertAlmostEqual(self.dlm3.result.predictStatus,
                               [11, 13, [-6.0/7, 6.0/7]])

    def testContinuePredictWithDynamic(self):
        self.dlm4.fitForwardFilter()
        featureDict = {'dynamic': 2.0}
        (obs, var) = self.dlm4.predict(date=9,
                                       featureDict=featureDict)
        self.assertAlmostEqual(self.dlm4.result.predictStatus,
                               [9, 10, [5.0/6 * 2]])

        featureDict = {'dynamic': 3.0}
        (obs, var) = self.dlm4.continuePredict(featureDict=featureDict)
        self.assertAlmostEqual(self.dlm4.result.predictStatus,
                               [9, 11, [5.0/6 * 2, 5.0/6 * 3]])

    def testPredictWithAutoReg(self):
        self.dlm5.fitForwardFilter()
        (obs, var) = self.dlm5.predict(date=99)
        self.assertAlmostEqual(obs, 100.03682874)
        (obs, var) = self.dlm5.continuePredict()
        self.assertAlmostEqual(obs, 101.07480945)

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
            comp.updateEvaluation(i)
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
            comp.updateEvaluation(i)
            trueAr[i] = comp.evaluation * trueAr[i]

        diff = 0.0
        for i in range(len(arTrend)):
            diff += abs(arTrend[i] - trueAr[i])
        self.assertAlmostEqual(diff, 0)

    def testGetVar(self):
        # for forward filter
        self.dlm5.fitForwardFilter()
        filteredTrend = self.dlm5.getVar(filterType='forwardFilter')
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
            comp.updateEvaluation(i)
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
            comp.updateEvaluation(i)
            trueAr[i] = comp.evaluation * trueAr[i] * comp.evaluation.T

        diff = 0.0
        for i in range(len(arTrend)):
            diff += abs(arTrend[i] - trueAr[i])
        self.assertAlmostEqual(diff, 0)

unittest.main()



#import numpy as np
#import pydlm as pd

#data = np.concatenate((np.random.random(100), np.random.random(100) + 3))
#myDLM = pd.dlm(data) + pd.trend(2, discount = 0.9)


#myDLM.turnOn('smooth')
#myDLM.turnOn('predict')
#myDLM.turnOff('multiple')
#myDLM.shrink(0.0)
#myDLM.fitForwardFilter(useRollingWindow = False, windowLength = 20)
#myDLM.fitForwardFilter()
#myDLM.fitBackwardSmoother()
#myDLM.plot()
