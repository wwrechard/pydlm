import unittest

from pydlm.modeler.trends import trend
from pydlm.modeler.autoReg import autoReg
from pydlm.func._dlmGet import _dlmGet


class test_dlmGet(unittest.TestCase):

    def setUp(self):
        self.data5 = range(100)
        self.dlm5 = _dlmGet(self.data5)
        self.dlm5.builder + trend(degree=0, discount=1, w=1.0) + \
            autoReg(degree=1, discount=1, w=1.0)
        self.dlm5._initialize()
        self.dlm5.options.innovationType='whole'

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
            comp.updateEvaluation(i, self.data5)
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
            comp.updateEvaluation(i, self.data5)
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
            comp.updateEvaluation(i, self.data5)
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
            comp.updateEvaluation(i, self.data5)
            trueAr[i] = comp.evaluation * trueAr[i] * comp.evaluation.T

        diff = 0.0
        for i in range(len(arTrend)):
            diff += abs(arTrend[i] - trueAr[i])
        self.assertAlmostEqual(diff, 0)

if __name__ == '__main__':
    unittest.main()
