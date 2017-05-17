[PyDLM](https://pydlm.github.io/)  [![Build Status](https://travis-ci.org/wwrechard/pydlm.svg?branch=master)](https://travis-ci.org/wwrechard/pydlm) [![Coverage Status](https://coveralls.io/repos/github/wwrechard/pydlm/badge.svg?branch=master)](https://coveralls.io/github/wwrechard/pydlm?branch=master)
=======================================================


Welcome to [pydlm](https://pydlm.github.io/), a flexible, user-friendly and rich functionality time series modeling library for python. This library implementes the Bayesian dynamic linear model (Harrison and West, 1999) for time series data. Time series modeling is easy with `pydlm`.

Updates in 0.1.1.8
------------------
* Add an modelTuner class to auto-tune the discounting factors using gradient descent.
* Add model evaluation methods for geting residuals and MSE (one-day a head predicted loss).
* Bug fix: Fix the incorrect return length of the DLM results.
* Add travis Build test and coverage test (Thanks @liguopku).
* Modify the tests on all Numpy matrix to pass Python3 tests.
* dynamic component now accepts Numpy Matrix as feature input (Thanks @xgdgsc).
* Update the doc to be more human readable (Thanks @xgdgsc).

What's next
-----------
* Extend to multi-variate DLM (Q3)
* Add more examples and template models with real world data (Q3)
* Support non-Gaussian noise and Evolutions via Sequential Monte Carlo (SMC) sampling (Q4)
* Refactor the code to be proto buffer based

Changes in the current Github (dev) version
-------------------------------------------
* Add `dlm.predictN()` which allows prediction over multiple days.

Installation
------------
You can currently get the package (version 0.1.1.8) from `pypi` by

      $ pip install pydlm

You can also get the latest from [github]
(https://github.com/wwrechard/PyDLM)

      $ git clone git@github.com:wwrechard/pydlm.git pydlm
      $ cd pydlm
      $ sudo python setup.py install

`pydlm` depends on the following modules,

* `numpy`     (for core functionality)
*  `matplotlib` (for plotting results)
* `Sphinx`    (for generating documentation)
* `unittest`  (for testing)

A simple example
-----------------
we give a simple example on linear regression to illustrate how to use the `pydlm` for analyzing data. The data is generated via the following process
```python
import numpy as np
n = 100
a = 1.0 + np.random.normal(0, 5, n) # the intercept
x = np.random.normal(0, 2, n) # the control variable
b = 3.0 # the coefficient
y = a + b * x
```
In the above code, `a` is the baseline random walk centered around 1.0 and `b` is the coefficient for a control variable. The goal is to
decompose `y` and learn the value of `a` and `b`. We first build the model
```python
from pydlm import dlm, trend, dynamic
mydlm = dlm(y)
mydlm = mydlm + trend(degree=1, discount=0.98, name='a', w=10.0)
mydlm = mydlm + dynamic(features=[[v] for v in x], discount=1, name='b', w=10.0)
```
In the model, we add two components `trend` and`dynamic`. The trend `a` is one of the systematical components that can be used to characterize the intrisic property of a time series, and trend is particularly suitable for our case. It has a discount factor of 0.98 as we believe the baseline can gradually shift overtime. The dynamic component `b` is modeling the regression component. We specify its discounting factor to be 1.0 since we believe `b` should be a constant. The `dynamic`class only accepts 2-d list for feature arugment (since the control variable could be multi-dimensional). Thus, we change `x` to 2d list. In addition, we believe these two processes `a` and `b` evolve independently and set (the following is currently the default assumption, so actually no need to set)
```python
mydlm.evolveMode('independent')
```
This can also be set to 'dependent' if the computation efficiency is a concern. The default prior on the covariance of each component is a
diagonal matrix with 1e7 on the diagonal and we changed this value in building the component by specifying `w` (more details please refer to the user manual). The prior on the observational noise (default to 1.0) can be set by
```python
mydlm.noisePrior(2.0)
```
We then fit the model by typing
```python
mydlm.fit()
```
After some information printed on the screen, we are done (yeah! :p) and we can fetch and examine our results. We can first visualize the fitted results and see how well the model fits the data
```python
mydlm.plot()
```
The result shows
<p align="center">
<img src="/doc/source/img/example_plot_all.png" width=80%/>
</p>

It looks pretty nice for the one-day ahead prediction accuracy. We can get the acumulated one-day ahead prediction loss by calling
```python
mydlm.getMSE()
```
and ask `mydlm` to tune the discounting factors to improve that metric
```python
mydlm.tune()
```
We can also plot the two coefficients `a` and `b` and see how they
change when more data is added
```python
mydlm.turnOff('predict')
mydlm.plotCoef(name='a')
mydlm.plotCoef(name='b')
```
and we have
<p align="center">
<img src="/doc/source/img/example_plot_a.png" width=49%/>
<img src="/doc/source/img/example_plot_b.png" width=49%/>
</p>

We see that the latent state of `b` quickly shift from 0 (which is our initial guess on the parameter) to around 3.0 and the confidence
interval explodes and then narrows down as more data is added.

Once we are happy about the result, we can fetch the results
```python
# get the smoothed time series
smoothedSeries = mydlm.getMean(filterType='backwardSmoother')
smoothedVar = mydlm.getVar(filterType='backwardSmoother')
smoothedCI = mydlm.getInterval(filterType='backwardSmoother')

# get the coefficients
coef_a = mydlm.getLatentState(filterType='backwardSmoother', name='a')
coef_a_var = mydlm.getLatentCov(filterType='backwardSmoother', name='a')
coef_b = mydlm.getLatentState(filterType='backwardSmoother', name='b')
coef_b_var = mydlm.getLatentCov(filterType='backwardSmoother', name='b')
```
We can then use `coef_a` and `coef_b` for further analysis. If we want to predict the future observation based on the current data, we
can do
```python
# prepare the new feature
newData1 = {'b': [5]}
# one-day ahead prediction from the last day
(predictMean, predictVar) = mydlm.predict(date=mydlm.n-1, featureDict=newData1)

# continue predicting for next day
newData2 = {'b': [4]}
(predictMean, predictVar) = mydlm.continuePredict(featureDict=newData2)

# continue predicting for the third day
newData3 = {'b': [3]}
(predictMean, predictVar) = mydlm.continuePredict(featureDict=newData3)
```
or using the simpler `dlm.predictN`
```python
newData = {'b': [[5], [4], [3]]}
(predictMean, predictVar) = mydlm.predictN(N=3, date=mydlm.n-1, featureDict=newData)
```

Documentation
-------------
Detailed documentation is provided in [PyDLM](https://pydlm.github.io/) with special attention to the [User manual](https://pydlm.github.io/#dynamic-linear-models-user-manual).

Changelogs
----------------
Updates in 0.1.1.7
(Special Thanks to Dr. Nick Gayeski for helping identify all these issues!)

* Add an option to let different component evolve independently (default)
* Bug fixing: change the default prior covariance for the components to match the results of BATS
* Bug fixing: deprecate the hand-written generalized inverse function and switch to numpy's built-in one.
* Add an easy specification for component prior on covariance and the model prior on observational noise (see the example and the user manual)

updates in 0.1.1.1

+ Fix bugs in latent states retrieval
+ Rewrite all the get methods (simpler and concise). Allows easy fetching individual component.
+ Add a longSeason component
+ Add more plot functionalities
+ Add the ribbon confidence interval
+ Add a simple example in documentation for using pydlm
