[PyDLM](https://pydlm.github.io/)  [![Build Status](https://travis-ci.org/wwrechard/pydlm.svg?branch=master)](https://travis-ci.org/wwrechard/pydlm) [![Coverage Status](https://coveralls.io/repos/github/wwrechard/pydlm/badge.svg?branch=master)](https://coveralls.io/github/wwrechard/pydlm?branch=master)
=======================================================


Welcome to [pydlm-lite](https://pydlm.github.io/), a flexible time series modeling library for python. This library is based on the Bayesian dynamic linear model (Harrison and West, 1999) and optimized for fast model fitting and inference.

This is the lite version of the main `pydlm` package where the plotting functionality and the dependency on the matplotlib has been removed. Most refactoring work will be conducted on this package to improve the class on multi-threading and online learning. In the meanwhile, the main `pydlm` package will remain in the current structure.

Going forward, the two packages will be developed towards two different directions

1. `pydlm` will support more sophisticated models and more advanced algorithm such as sequential monte carlo. The algorithm will be optimized in terms of accuracy rather than latency. The primary use case is on advanced inference and data analysis with small datasets.

2. `pydlm-lite` will mainly focus on normal-normal and poisson-gamma models with the fastest possible fitting algorithm. The class design will support concurrency and online updating. The primary use case is scalable anomaly detection and forecasting with millions of time series.

Updates in the github version
-------------------------------------------
* Plan to refactor the `dlm` class. Align with the goal to separate model and data, I'm going to refactor the `dlm` class such that
  1. The main `dlm` class will only contain model build information and is supposed to be 'const' after construction.
  2. Time series data will be passed in as an argument to the `fit` or `forwardFilter` and the fitted result will be returned as well as the model status.
  3. Model status can also be passed into `fit` and `forwardFilter` as a prior.
  The goal is to make the `dlm` class state-independent, so that the class is thread-safe and can be shared by multiple threads for parallel processing. While in progress, all the old class behavior will be kept.

Installation
------------
You can get the package (current version 0.1.1.10) from `pypi` by

      $ pip install pydlm-lite

You can also get the latest from [github](https://github.com/wwrechard/PyDLM)

      $ git clone git@github.com:wwrechard/pydlm.git pydlm
      $ cd pydlm
      $ sudo python setup.py install

`pydlm` depends on the following modules,

* `numpy`     (for core functionality)
* `Sphinx`    (for generating documentation)
* `unittest`  (for testing)

Google data science post example
-----------------
We use the example from the [Google data science post](http://www.unofficialgoogledatascience.com/2017/07/fitting-bayesian-structural-time-series.html) as an example to show how `pydlm` could be used to analyze the real world data. The code and data is placed under `examples/unemployment_insurance/...`. The dataset contains weekly counts of initial claims for unemployment during 2004 - 2012 and is available from the R package `bsts` (which is a popular R package for time series modeling). The raw data is shown below (left)
<p align="center">
<img src="/doc/source/img/unemployment_2.png" width=48%/>
<img src="/doc/source/img/unemployment_1.png" width=48%/>
</p>
We see strong annual pattern and some local trend from the data.
<h4> A simple model </h4>
Following the Google's post, we first build a simple model with only local linear trend and seasonality component.

```python
from pydlm import dlm, trend, seasonality
# A linear trend
linear_trend = trend(degree=1, discount=0.95, name='linear_trend', w=10)
# A seasonality
seasonal52 = seasonality(period=52, discount=0.99, name='seasonal52', w=10)
# Build a simple dlm
simple_dlm = dlm(time_series) + linear_trend + seasonal52
```

In the actual code, the time series data is scored in the variable `time_series`. `degree=1` indicates the trend is linear (2 stands for quadratic) and `period=52` means the seasonality has a periodicy of 52. Since the seasonality is generally more stable, we set its discount factor to 0.99. For local linear trend, we use 0.95 to allow for some flexibility. `w=10` is the prior guess on the variance of each component, the larger number the more uncertain. For actual meaning of these parameters, please refer to the [user manual](https://pydlm.github.io/#dynamic-linear-models-user-manual). After the model built, we can fit the model and plot the result (shown above, right figure)

```python
# Fit the model
simple_dlm.fit()
```

The blue curve is the forward filtering result, the green curve is the one-day ahead prediction and the red curve is the backward smoothed result. The light-colored ribbon around the curve is the confidence interval (you might need to zoom-in to see it). The one-day ahead prediction shows this simple model captures the time series somewhat good but loses accuracy around the peak crisis at Week 280 (which is between year 2008 - 2009). The one-day-ahead mean squared prediction error is **0.173** which can be obtained by calling

```python
simple_dlm.getMSE()
```

We can decompose the time series into each of its components

```python
simple_dlm.getMean(filterType='backwardSmoother', name='linear_trend')
simple_dlm.getMean(filterType='backwardSmoother', name='seasonal52')
```

<p align="center">
<img src="/doc/source/img/unemployment_3-1.png" width=49%/>
<img src="/doc/source/img/unemployment_4-1.png" width=49%/>
</p>

Most of the time series shape is attributed to the local linear trend and the strong seasonality pattern is easily seen. To further verify the performance, we use this simple model for long-term forecasting. In particular, we use the previous **351 week**'s data to forecast the next **200 weeks** and the previous **251 week**'s data to forecast the next **200 weeks**. We lay the predicted results on top of the real data

```python
# predict given the first 351 weeks and forcast the next 200 weeks.
simple_dlm.predictN(date=350, N=200)
# predict given the first 251 weeks and forcast the next 200 weeks.
simple_dlm.predictN(date=250, N=200)
```

<p align="center">
<img src="/doc/source/img/unemployment_5.png" width=49%/>
<img src="/doc/source/img/unemployment_6.png" width=49%/>
</p>

From the figure we see that after the crisis peak around 2008 - 2009 (Week 280), the simple model can accurately forecast the next 200 weeks (left figure) given the first 351 weeks. However, the model fails to capture the change near the peak if the forecasting start before Week 280 (right figure).

<h4> Dynamic linear regression </h4>
Now we build a more sophiscated model with extra variables in the data file. The extra variables are stored in the variable `features` in the actual code. To build the dynamic linear regression model, we simply add a new component

```python
# Build a dynamic regression model
from pydlm import dynamic
regressor10 = dynamic(features=features, discount=1.0, name='regressor10', w=10)
drm = dlm(time_series) + linear_trend + seasonal52 + regressor10
drm.fit()
drm.getMSE()
```

`dynamic` is the component for modeling dynamically changing predictors, which accepts `features` as its argument. The above code plots the fitted result (top left).

<p align="center">
<img src="/doc/source/img/unemployment_7.png" width=48%/>
<img src="/doc/source/img/unemployment_8-1.png" width=48%/>
<img src="/doc/source/img/unemployment_9-1.png" width=48%/>
<img src="/doc/source/img/unemployment_10-1.png" width=48%/>
</p>

The one-day ahead prediction looks much better than the simple model, particularly around the crisis peak. The mean prediction error is **0.099** which is a 100% improvement over the simple model. Similarly, we also decompose the time series into the three components

```python
drm.getMean(filterType='backwardSmoother', name='linear_trend')
drm.getMean(filterType='backwardSmoother', name='seasonal52')
drm.plot('regressor10')
```

This time, the shape of the time series is mostly attributed to the regressor and the linear trend looks more linear. If we do long-term forecasting again, i.e., use the previous **301 week**'s data to forecast the next **150 weeks** and the previous **251 week**'s data to forecast the next **200 weeks**

```python
drm.plotPredictN(date=300, N=150)
drm.plotPredictN(date=250, N=200)
```

<p align="center">
<img src="/doc/source/img/unemployment_11.png" width=48%/>
<img src="/doc/source/img/unemployment_12.png" width=48%/>
</p>

The results look much better compared to the simple model

Documentation
-------------
Detailed documentation is provided in [PyDLM](https://pydlm.github.io/) with special attention to the [User manual](https://pydlm.github.io/#dynamic-linear-models-user-manual).
