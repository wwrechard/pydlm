import unittest

from pydlm.modeler.trends import trend
from pydlm.dlm import dlm

class testDlmPlot(unittest.TestCase):

    def testPlot(self):
        dlm1 = dlm(range(100)) + trend(1)
        dlm1.fit()
        dlm1.plot()

if __name__ == '__main__':
    unittest.main()
