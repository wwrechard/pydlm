import sys
import numpy as np

sys.path.append('/Users/samuel/Documents/Github/PyDLM/src/')

from pydlm.modeler.trends import trend
from pydlm.func.dlm_base import dlm_base

data = [0] * 9 + [1] + [0] * 10
dlm1 = dlm_base(data)
trend1 = trend(degree = 1, discount = 1)
dlm1.builder.add(trend1)

dlm1.__initialize__()
dlm1.__forwardFilter__(start = 0, end = 19)
dlm1.__backwardSmoother__(start = 19)
