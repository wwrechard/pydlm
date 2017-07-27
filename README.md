[PyDLM](https://pydlm.github.io/)  [![Build Status](https://travis-ci.org/wwrechard/pydlm.svg?branch=master)](https://travis-ci.org/wwrechard/pydlm) [![Coverage Status](https://coveralls.io/repos/github/wwrechard/pydlm/badge.svg?branch=master)](https://coveralls.io/github/wwrechard/pydlm?branch=master)
=======================================================


Welcome to [pydlm](https://pydlm.github.io/), a flexible, user-friendly and rich functionality time series modeling library for python. This library implementes the Bayesian dynamic linear model (Harrison and West, 1999) for time series data. Time series modeling is easy with `pydlm`.

Changes in the current Github (dev) version
-------------------------------------------
* Add an example from Google data science blog (Soon to update the readme page with the new example)
* Add `dlm.plotPredictN()` which plots the prediction result from `dlm.predictN()` on top of the time series data.
* Add `dlm.predictN()` which allows prediction over multiple days.
* Change the `degree` of `trend` to match the actual meaning in polynomial, i.e, `degree=0` stands for constant and `degree=1` stands for linear trend and so on so forth.
* Add support for missing data in `modelTuner` and `.getMSE()` (Thanks @sun137653577)

Installation
------------
You can currently get the package (version 0.1.1.8) from `pypi` by

      $ pip install pydlm

You can also get the latest from [github](https://github.com/wwrechard/PyDLM)

      $ git clone git@github.com:wwrechard/pydlm.git pydlm
      $ cd pydlm
      $ sudo python setup.py install

`pydlm` depends on the following modules,

* `numpy`     (for core functionality)
*  `matplotlib` (for plotting results)
* `Sphinx`    (for generating documentation)
* `unittest`  (for testing)

Google data science post example
-----------------
We use the example from the [Google data science post](http://www.unofficialgoogledatascience.com/2017/07/fitting-bayesian-structural-time-series.html) to showcase `pydlm`. The code and data is placed under `examples/unemployment_insurance/...`. The data is the weekly counts of initial claims for unemployment during 2004 - 2012 and is available from the R-package `bsts` (which is another popular R package for time series modeling). The raw data is shown below (left)
<p align="center">
<img src="/doc/source/img/unemployment_2.png" width=48%/>
<img src="/doc/source/img/unemployment_1.png" width=48%/>
</p>
We can see strong annual pattern and some local trend from the data.
<h4> A simple model </h4>
Following the post, we first build a simple model with only local linear trend and seasonality component.

```python
from pydlm import dlm, trend, seasonality
# A linear trend
linear_trend = trend(degree=1, discount=0.95, name='linear_trend', w=10)
# A seasonality
seasonal52 = seasonality(period=52, discount=0.99, name='seasonal52', w=10)
# Build a simple dlm
simple_dlm = dlm(time_series) + linear_trend + seasonal52
```

In the actual code, the time series data is scored in the variable `time_series`. `degree=1` indicates the trend is a linear (2 stands for quadratic) and `period=52` means the seasonality has a periodicy of 52. Usually, the seasonality is more stable, so we set the discount factor to 0.99 for seasonality and 0.95 for linear trend to allow some flexibility. `w=10` is the prior guess on the variance of the component, the larger the uncertain. For actual meaning of these parameters, please refer to the [user manual](https://pydlm.github.io/#dynamic-linear-models-user-manual). After the model built, we can fit the model and plot the result (shown above, right figure)

```python
# Fit the model
simple_dlm.fit()
# Plot the fitted results
simple_dlm.turnOff('data points')
simple_dlm.plot()
```

The blue curve is the forward filtering result, the green curve is the one-day ahead prediction and the red curve is the backward smoothed result. The light-colored ribbon around the curve is the confidence interval (you might need to zoom-in to see it). The one-day ahead prediction shows this simple model captures the time series somewhat good but loses track around the peak at Week 280 (which is between year 2008 - 2009). The one-day ahead mean prediction error is **0.173*

```python
simple_dlm.getMSE()
```

We can also decompose the time series to each of its component

```python
# Plot each component (attribute the time series to each component)
simple_dlm.turnOff('predict plot')
simple_dlm.turnOff('filtered plot')
simple_dlm.plot('linear_trend')
simple_dlm.plot('seasonal52')
```

<p align="center">
<img src="/doc/source/img/unemployment_3-1.png" width=49%/>
<img src="/doc/source/img/unemployment_4-1.png" width=49%/>
</p>

Most of the time series shape is attributed to the local linear trend and the strong seasonality pattern is easily seen. To further verify the performance, we use this simple model for long-term forecasting. In particular, we use the previous **351 week**'s data to forecast the next **200 weeks** and the previous **251 week**'s data to forecast the next **200 weeks**. We lay the predicted results over the real data

```python
# Plot the prediction give the first 351 weeks and forcast the next 200 weeks.
simple_dlm.plotPredictN(date=350, N=200)
# Plot the prediction give the first 251 weeks and forcast the next 200 weeks.
simple_dlm.plotPredictN(date=250, N=200)
```

<p align="center">
<img src="/doc/source/img/unemployment_5.png" width=49%/>
<img src="/doc/source/img/unemployment_6.png" width=49%/>
</p>

From the figure we see that, after the crisis peak around 2008 - 2009 (Week 280), the simple model can accurately forecast the next 200 weeks (left figure) given the first 351 weeks. However, the model fails to capture the change near the peak if the forecasting start before Week 280 (right figure).

<h4> Dynamic linear regression </h4>
Now we build a more sophiscated model with extra variables in the data file. The extra variables are stored in the variable `features` in the actual code. To build the dynamic linear regression model, we simply add a new component

```python
# Build a dynamic regression model
from pydlm import dynamic
regressor10 = dynamic(features=features, discount=1.0, name='regressor10', w=10)
drm = dlm(time_series) + linear_trend + seasonal52 + regressor10
drm.fit()
drm.getMSE()

# Plot the fitted results
drm.turnOff('data points')
drm.plot()
```

`dynamic` is the component for dynamically changed control variables, which accepts `features` as its argument. The above code plots the fitted result (top left)

<p align="center">
<img src="/doc/source/img/unemployment_7.png" width=48%/>
<img src="/doc/source/img/unemployment_8-1.png" width=48%/>
<img src="/doc/source/img/unemployment_9-1.png" width=48%/>
<img src="/doc/source/img/unemployment_10-1.png" width=48%/>
</p>

and the one-day ahead prediction curve looks much better than the simple model, particularly around the crisis peak. The mean prediction error is **0.099** which is a 100% improvement over the simple model. Similarly, we also decompose the time series to the three components

```python
drm.turnOff('predict plot')
drm.turnOff('filtered plot')
drm.plot('linear_trend')
drm.plot('seasonal52')
drm.plot('regressor10')
```

The results are shown in the above fitures. This time, the shape of the time series is mostly attributed to the regressor and the linear trend looks more linear. If we do long-term forecasting again, i.e., use the previous **301 week**'s data to forecast the next **150 weeks** and the previous **251 week**'s data to forecast the next **200 weeks**

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
