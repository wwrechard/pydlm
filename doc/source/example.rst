.. currentmodule:: pydlm

In this section, we give a simple example on linear regression to
illustrate how to use the `pydlm` for analyzing the data. The data is
generated via the following process::

  >>> import numpy as np
  >>> n = 100
  >>> a = 1.0 + np.random.normal(0, 5, n) # the intercept
  >>> x = np.random.normal(0, 2, n) # the control variable
  >>> b = 3.0 # the coefficient
  >>> y = a + b * x

In the above code, `a` is the baseline random walk centered around 1.0
and `b` is the coefficient for a control variable. The goal is to
decompose `y` and learn the value of `a` and `b`. We first build the
model::

  >>> from pydlm import dlm, trend, dynamic
  >>> mydlm = dlm(y)
  >>> mydlm = mydlm + trend(degree=1, discount=0.98, name='a')
  >>> mydlm = mydlm + dynamic(features=[[i] for i in x], discount=1, name='b')

In the model, we add two components :class:`trend` and
:class:`dynamic`. The trend `a` is one kind of systematical components
that can be used to characterize the intrisic property of a time
series, and trend is particularly suitable in our case. The dynamic
component `b` is modeling the regression component. We specify its
discounting factor to be 1.0 means that we believe `b` should be a
constant, while for `a` we use 0.98 as we believe baseline can be
gradually shift overtime. Then we fit the model::

  >>> mydlm.fit()

After some information printed by the system, we are done in model
fitting (yeah! :p) and we can fetch and examine our results. We can
first visualize the fitted results and see how well we fit the data::

  >>> mydlm.plot()

We can also plot the two coefficients `a` and `b` and see how they
change when more data come in::

  >>> mydlm.turnOff('predict')
  >>> mydlm.plotCoef(name='a')
  >>> mydlm.plotCoef(name='b')

We see that the latent state of `b` quickly shift from 0, which is our
initial guess on the parameter to around 3.0 and the confidence
interval explode and then narrow down as more data comes.

Once we are happy about the result, we can fetch the results:::

  >>> # get the smoothed results
  >>> smoothedResult = mydlm.getMean(filterType='backwardSmoother')
  >>> smoothedVar = mydlm.getVar(filterType='backwardSmoother')
  >>> smoothedCI = mydlm.getInterval(filterType='backwardSmoother')
  >>>
  >>> # get the coefficients
  >>> coef_a = mydlm.getLatentState(filterType='backwardSmoother', name='a')
  >>> coef_a_var = mydlm.getLatentCov(filterType='backwardSmoother', name='a')
  >>> coef_b = mydlm.getLatentState(filterType='backwardSmoother', name='b')
  >>> coef_b_var = mydlm.getLatentCov(filterType='backwardSmoother', name='b')

We can then use `coef_a` and `coef_b` for further analysis.





