import numpy as np

from pydlm.modeler.builder import builder
from pydlm.modeler.trends import trend
from pydlm.modeler.seasonality import seasonality
from pydlm.modeler.dynamic import dynamic
from pydlm.modeler.matrixTools import matrixTools as mt
class testBuilder:

    def __init__(self):
        self.features = np.matrix(np.random.rand(2, 10))
        self.trendD = 3
        self.seasonalityD = 7
        self.tol = 0.001

        self.trend = trend(self.trendD)
        self.seasonality = seasonality(self.seasonalityD)
        self.dynamic = dynamic(self.features)

    def testRun(self):
        self.testInitialization()
        self.testAddAndDelete()
        self.testInitialize()
        self.testUpdate()
        
    def testInitialization(self):
        builder1 = builder()
        assert builder1.step == 0
        assert len(builder1.dynamicComponents) == 0
        pass

    def testAddAndDelete(self):
        builder1 = builder()
        builder1 = builder1 + self.trend
        assert len(builder1.staticComponents) == 1
        assert len(builder1.dynamicComponents) == 0

        builder1 = builder1 + self.dynamic
        assert len(builder1.staticComponents) == 1
        assert len(builder1.dynamicComponents) == 1

        builder1 = builder1 + self.seasonality
        assert len(builder1.staticComponents) == 2
        assert len(builder1.dynamicComponents) == 1

        builder1.delete('seasonality')
        assert len(builder1.staticComponents) == 1
        assert len(builder1.dynamicComponents) == 1

        assert builder1.staticComponents['trend'] == self.trend
        pass
    
    def testInitialize(self):
        builder1 = builder()
        builder1 = builder1 + self.trend + self.dynamic
        
        builder1.initialize()
        assert np.sum(np.abs(builder1.model.evaluation - mt.matrixAddByCol(self.trend.evaluation, self.dynamic.evaluation))) < self.tol
        pass

    def testUpdate(self):
        builder1 = builder()
        builder1 = builder1 + self.trend + self.dynamic     
        builder1.initialize()

        builder1.updateEvaluation(2)
        assert np.sum(np.abs(builder1.model.evaluation - mt.matrixAddByCol(self.trend.evaluation, self.features[:, 2].T))) < self.tol
        pass

aTest = testBuilder()
aTest.testRun()
