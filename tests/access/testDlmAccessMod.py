import numpy as np
import unittest

from pydlm.modeler.trends import trend
from pydlm.modeler.autoReg import autoReg
from pydlm.access.dlmAccessMod import dlmAccessModule


class testDlmAccessMod(unittest.TestCase):
    def setUp(self):
        self.data5 = range(100)
        self.dlm5 = dlmAccessModule(self.data5)
        (
            self.dlm5.builder
            + trend(degree=0, discount=1, w=1.0)
            + autoReg(degree=1, discount=1, w=1.0)
        )
        self.dlm5._initialize()
        self.dlm5.options.innovationType = "whole"

    def testGetMean(self):
        # for forward filter
        self.dlm5._forwardFilter(start=0, end=99, renew=False)
        self.dlm5.result.filteredSteps = [0, 99]
        filteredTrend = self.dlm5.getMean(filterType="forwardFilter")
        self.assertEqual(len(filteredTrend), self.dlm5.n)
        np.testing.assert_array_equal(
            filteredTrend, self.dlm5._1DmatrixToArray(self.dlm5.result.filteredObs)
        )

        # for predict filter
        predictedTrend = self.dlm5.getMean(filterType="predict")
        np.testing.assert_array_equal(
            predictedTrend, self.dlm5._1DmatrixToArray(self.dlm5.result.predictedObs)
        )

        # for component with forward filter
        arTrend = self.dlm5.getMean(filterType="forwardFilter", name="ar2")
        trueAr = [item[1, 0] for item in self.dlm5.result.filteredState]
        comp = self.dlm5.builder.automaticComponents["ar2"]
        for i in range(len(trueAr)):
            comp.updateEvaluation(i, self.data5)
            trueAr[i] = comp.evaluation * trueAr[i]
        np.testing.assert_array_equal(arTrend, self.dlm5._1DmatrixToArray(trueAr))

        # for backward smoother
        self.dlm5._backwardSmoother(start=99)
        self.dlm5.result.smoothedSteps = [0, 99]
        filteredTrend = self.dlm5.getMean(filterType="backwardSmoother")
        np.testing.assert_array_equal(
            filteredTrend, self.dlm5._1DmatrixToArray(self.dlm5.result.smoothedObs)
        )

        # for component with backward smoother
        arTrend = self.dlm5.getMean(filterType="backwardSmoother", name="ar2")
        trueAr = [item[1, 0] for item in self.dlm5.result.smoothedState]
        comp = self.dlm5.builder.automaticComponents["ar2"]
        for i in range(len(trueAr)):
            comp.updateEvaluation(i, self.data5)
            trueAr[i] = comp.evaluation * trueAr[i]

        np.testing.assert_array_equal(arTrend, self.dlm5._1DmatrixToArray(trueAr))

    def testGetVar(self):
        # for forward filter
        self.dlm5._forwardFilter(start=0, end=99, renew=False)
        self.dlm5.result.filteredSteps = [0, 99]

        filteredTrend = self.dlm5.getVar(filterType="forwardFilter")
        self.assertEqual(len(filteredTrend), self.dlm5.n)
        np.testing.assert_array_equal(
            filteredTrend, self.dlm5._1DmatrixToArray(self.dlm5.result.filteredObsVar)
        )

        # for predict filter
        predictedTrend = self.dlm5.getVar(filterType="predict")
        np.testing.assert_array_equal(
            predictedTrend, self.dlm5._1DmatrixToArray(self.dlm5.result.predictedObsVar)
        )

        # for component with forward filter
        arTrend = self.dlm5.getVar(filterType="forwardFilter", name="ar2")
        trueAr = [item[1, 1] for item in self.dlm5.result.filteredCov]
        comp = self.dlm5.builder.automaticComponents["ar2"]
        for i in range(len(trueAr)):
            comp.updateEvaluation(i, self.data5)
            trueAr[i] = comp.evaluation * trueAr[i] * comp.evaluation.T

        np.testing.assert_array_equal(arTrend, self.dlm5._1DmatrixToArray(trueAr))

        # for backward smoother
        self.dlm5._backwardSmoother(start=99)
        self.dlm5.result.smoothedSteps = [0, 99]

        filteredTrend = self.dlm5.getVar(filterType="backwardSmoother")
        np.testing.assert_array_equal(
            filteredTrend, self.dlm5._1DmatrixToArray(self.dlm5.result.smoothedObsVar)
        )

        # for component with backward smoother
        arTrend = self.dlm5.getVar(filterType="backwardSmoother", name="ar2")
        trueAr = [item[1, 1] for item in self.dlm5.result.smoothedCov]
        comp = self.dlm5.builder.automaticComponents["ar2"]
        for i in range(len(trueAr)):
            comp.updateEvaluation(i, self.data5)
            trueAr[i] = comp.evaluation * trueAr[i] * comp.evaluation.T
        np.testing.assert_array_equal(arTrend, self.dlm5._1DmatrixToArray(trueAr))

    def testGetLatentState(self):
        # for forward filter
        self.dlm5._forwardFilter(start=0, end=99, renew=False)
        self.dlm5.result.filteredSteps = [0, 99]

        filteredTrend = self.dlm5.getLatentState(filterType="forwardFilter")
        for i in range(100):
            np.testing.assert_array_equal(
                filteredTrend[i], self.dlm5.result.filteredState[i].flatten().tolist()
            )

        # for predict filter
        predictedTrend = self.dlm5.getLatentState(filterType="predict")
        for i in range(100):
            np.testing.assert_array_equal(
                predictedTrend[i], self.dlm5.result.predictedState[i].flatten().tolist()
            )

        # for backward smoother
        self.dlm5._backwardSmoother(start=99)
        self.dlm5.result.smoothedSteps = [0, 99]

        smoothedTrend = self.dlm5.getLatentState(filterType="backwardSmoother")
        for i in range(100):
            np.testing.assert_array_equal(
                smoothedTrend[i], self.dlm5.result.smoothedState[i].flatten().tolist()
            )

    def testGetLatentCov(self):
        # for forward filter
        self.dlm5._forwardFilter(start=0, end=99, renew=False)
        self.dlm5.result.filteredSteps = [0, 99]

        filteredTrend = self.dlm5.getLatentCov(filterType="forwardFilter")
        np.testing.assert_array_equal(filteredTrend, self.dlm5.result.filteredCov)

        # for predict filter
        predictedTrend = self.dlm5.getLatentCov(filterType="predict")
        np.testing.assert_array_equal(predictedTrend, self.dlm5.result.predictedCov)

        # for backward smoother
        self.dlm5._backwardSmoother(start=99)
        self.dlm5.result.smoothedSteps = [0, 99]

        smoothedTrend = self.dlm5.getLatentCov(filterType="backwardSmoother")
        np.testing.assert_array_equal(smoothedTrend, self.dlm5.result.smoothedCov)


if __name__ == "__main__":
    unittest.main()
