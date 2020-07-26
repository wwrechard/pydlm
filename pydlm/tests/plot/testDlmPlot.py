import unittest

from pydlm.dlm import dlm
from pydlm.modeler.trends import trend


class testDlmPlot(unittest.TestCase):

    def testPlot(self):
        dlm1 = dlm(range(100)) + trend(1)
        dlm1.fit()
        dlm1.plot()


if __name__ == '__main__':
    unittest.main()
