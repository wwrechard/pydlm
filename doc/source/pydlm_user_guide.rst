.. currentmodule:: pydlm

This package implements the Bayesian dynamic linear model (DLM, Harrison
and West, 1999) for time series analysis.
The DLM is built upon two layers. The first layer is
the fitting algorithm. DLM adopts a modified Kalman filter with a
unique discounting technique from
Harrison and West (1999). Like the usual Kalman filter, it accepts a
transition matrix, a measurement matrix, an observation, a latent
state, an innovation and an error covariance matrix and return the
updated state and error covariance. These quantities will all be
supplied internally -- users are free from any annoying
calculations. Different from the usual Kalman filter, the modified
Kalman filter does not require the tuning of the two parameters: the
error covariance matrix and the observational variance, so the model
fitting is extremely efficient (could be up to 1000 times faster than
the EM algorithm), more details will be provided in the section of the
discounting technique.

The second layer of DLM is itsmodeling feature. Nicely summarized in
Harrison and West (1999), most common models can be expressed in
one unified form -- canonical form, which is closely related to the
Jordan decomposition. Thanks to this keen observation, the DLM can
easily incorporate most modeling components and turn them into the
corresponding transition matrices and other quantities to be supplied
to the Kalman filter. Examples are trend, seasonality, holidays,
control variables and auto-regressive, which could appear
simultaneously in one model. Due to this nice property, users of this
package can construct models simply by "adding" some component into
the model as::

  >>> myDLM = dlm(data) + trend(2)

The modeling process is simple.

The purpose of the modeling is to better understand the time series
data and for to forecast into the future. So the key output from the
model are the filtered time series, smoothed time series and one-step
ahead prediction. We will cover this topic later in this section.

The advantage of `pydlm`:

    + flexibility in constructing complicated models

    + Extremely efficient model fitting with the discounting technique

    + user-specific adjustment on the adaptive property of the model

The disadvantage of `pydlm`:

    + only for Gaussian noise


Modeling
========

As discussed in the beginning, the modeling process is very simple
with `pydlm`, most modeling functions are integrated in the class
:class:`dlm`. Following is an example for constructing a dlm with
linear trend, 7-day seasonality and another control variable::

  >>> from pydlm import dlm, trend, seasonality, dynamic
  >>> data = [0] * 100 + [3] * 100
  >>> control = [0.5 for i in range(100) + [2.6 for i in
  range(100)]
  >>> myDLM = dlm(data) + trend(2) + seasonality(7) +
  dyanmic(control)

The imput :attr:`data` must an 1d array or a list, since the current
:class:`dlm` only supports one dimensional time series. Supporting for
multivariate time series will be built upon this one dimensional class
and added in the future.

Now the variable `myDLM` contains the data and the modeling
information. It will construct the corresponding transition,
measurement, innovation, latent states and error covariance matrix
once model fitting is called. Modify an existing model is also
simple. User can brows the existing components of the model by::

  >>> myDLM.ls()

It will show all the existing components and their corresponding
names. Name can be specified when the component is added to the `dlm`,
for example::

  >>> myDLM = myDLM + seasonality(4, name = 'day4')
  >>> myDLM.ls()

We can also easily delete the unwanted component by using `delete`::

  >>> myDLM.delete('day4')


Model components
===================

There are four model components provided with this
package: trend, seasonality, dynamic and the auto-regression.

Trend
-----
:class:`trend` class is a model component for trending
behavior. The data might be increasing linearly or quadraticly, which
can all be captured by :class:`trend`. The degree argument specifics
the shape of the trend and the discounting factor will be explained
later in next section::

  >>> linearTrend = trend(degree = 2, discount = 0.99, name = 'trend1')

Seasonality
-----------
The :class:`seasonality` class models the periodic behavior of the
data. Compared to the sine or cosine periodic curves,
:class:`seasonality` in this packages is more flexible, since it can
turn into any shapes, much broader than the triangular families::

  >>> weekPeriod = seasonality(period = 7, discount = 0.99, name = 'week')

In the package, we implements the seasonality component in a
`form-free` way (Harrison and West, 1999) to avoid the identifiability
issue. The states of one seasonality component are always summed up to
zero, so that it will not tangle with the :class:`trend` component.
  
Dynamic
-------
The :class:`dynamic` class offers the modeling ability to add any additional
observed time series as a controlled variable to the current one. For
example, when studying the stock price, the 'SP500' index could be a
good indicator for the modeling stock. A dynamic component need the
user to supply the necessary information of the control variable over
time::

  >>> SP500 = dynamic(features = SP500Index, discount = 0.99, name =
  'SP500')

The input :attr:`features` for :class:`dynamic` should be a list of
lists, since multi-dimension features are allowed. Following is one
simple example::

  >>> Features = [[1.0, 2.0], [1.0, 3.0], [3.0, 3.0]]


Auto-regression
---------------
The :class:`autoReg` class constructs the auto-regressive component on
the model, i.e., the direct linear or non-linear dependency between
the current observation and the previous days. User needs to specify
the number of days of the dependency::

  >>> AR3 = autoReg(degree = 3, discount = 0.99, name = 'ar3')

These four classes of model components offer abundant modeling
possiblities of the Bayesian dynamic linear model. Users can construct
very complicated models using these components, such as hourly, weekly or
monthly periodicy and holiday indicator and many other features.


Model fitting
=============

Entailed before, the fitting of the dlm is fulfilled by a modified
Kalman filter. Once the user finished constructing the model by adding
different components. the :class:`dlm` will compute all the necessary
quantities internally for using Kalman filter. So users can simply
call :func:`dlm.fitForwardFilter`, :func:`dlm.fitBackwardSmoother` or
even simply :func:`dlm.fit` to fit both forward filter and backward
smoother::

  >>> myDLM.fitForwardFilter()
  >>> myDLM.fitBackwardSmoother()
  >>> myDLM.fit()

The :func:`dlm.fitForwardFilter` is implemented in an online
manner. It keeps an internal count on the filtered dates and once new
data comes in, it only filters the new data without touching the
existing results. In addition, this function also allows a rolling
window fitting on the data, i.e., there will be a moving window and
for each date, the kalman filter will only use the data within the
window to filter the observation. This is equivalent to that the model
only remembers a fixed length of dates::

  >>> myDLM.fitForwardFilter(useRollingWindow = True, windowLength
  = 30)

For :func:`dlm.backwardSmoother`, it has to use the whole time series
to smooth the latent states once new data comes in. The smoothing
provides a good retrospective analysis on our past decision of the
data. For example, we might initially believe the time series is
stable, while that could be a random behavior within a volatile time
series, and the user learn this from the smoother.

Once the model fitting is completed, users can fetch the filtered or
smoothed results from :class:`dlm`::

  >>> myDLM.getFilteredObs()
  >>> myDLM.getSmoothedObs()
  >>> myDLM.getFilteredVar()

The :class:`dlm` recomputes a wide variety of model quantities that
can be extracted by the user. For example, user can get the filtered
states and covariance by typing::

  >>> myDLM.getFilteredState()
  >>> myDLM.getFilteredCov()

This can be specified into individual component. For example, assume the
model contains a :class:`trend` component with a name of `trend1`, we
can extract the corresponding latent state only for `trend1` as::

  >>> myDLM.getFilteredState(name = 'trend1')
  >>> myDLM.getFilteredCov(name = 'trend1')

One can also get the confidence interval on the filtered time series::

  >>> myDLM.getFilteredInterval(p = 0.99)

There are also corresponding methods for smoothed and predicted
results. For more detail, please refer to the class documentation.


Model amending
==============

The user can still add, delete, modify data even when the model has
been constructed.

Adding new data
---------------
For adding more data, user can opt to
:func:`dlm.append`::

   >>> newData = [0, 1, 2]
   >>> myDLM.append(newData, component = 'main')

If the model contains :class:`dynamic` component, the corresponding
features need to be updated as well::

  >>> newSP500 = [[2000], [2100], [2200]]
  >>> myDLM.append(data = newSP500, component = 'SP500')

Then the user can rerun the forward filter::

  >>> myDLM.fitForwardFilter()

The package will continue running the forward filter on the three new
data ponts.

Deleting existing data
----------------------
To delete any existing data, user can simply use the :func:`dlm.popout`
function from :class:`dlm` on a specific date, for example::

  >>> myDLM.popout(1)

Different from :func:`dlm.append`, :func:`dlm.popout` will be executed
automatically for all components, so the user does not need to conduct
the deletion mannually for each component. After the deletion, the
forward filter needs to be rerun following the deleted date::

  >>> myDLM.fitForwardFilter()

Again, the package will automatically recognize the date and fit only
the necessary period of time.

Ignoring a date
---------------
Ignoring is very similar to deleting. The only difference is the time
counts. Because deleting will delete the data entirely, the time
counts will therefore reduce by 1. By contrast, ignoring will treat
the specific date as missing data, so the time count will not
change. This difference is important when preriodicy is
concerned. Changing of time counts will have high impacts on
:class:`seasonality` components.

:func:`dlm.ignore` simply set the data of a specific date to be None::
	
  >>> myDLM.ignore(2)

modify data
-----------
The :class:`dlm` also provides user the ability to modify the data on a
specific date and a specific component. This function enables possible
future extension to interactive anomaly detection and data debugging::

  >>> myDLM.alter(date = 2, data = 0, component = 'main')


Model plotting
==============

This package offers rich ploting options for illustrating the
results. User can simply call :func:`dlm.plot` to directly plot the
results once the models are fitted::

  >>> myDLM.plot()

User can choose which results to plot via :func:`dlm.turnOn` and
:func:`dlm.turnOff`::

   >>> myDLM.turnOn('filtered plot')
   >>> myDLM.turnOff('predict plot')
   >>> myDLM.turnOff('smoothed plot')

User can also choose whether to plot the confidence interval and
whether plot the results in one figure or separate figures. The
default is to plot the confidence interval and in separate plots. To
change that::

  >>> myDLM.turnOff('confidence')
  >>> myDLM.turnOff('multiple plots')

The quantile of the confidence interval can be set via
:func:`dlm.setConfidence`::

  >>> myDLM.setConfidence(p = 0.95)

The default colors for the plots are:

    + original data: black
    + filtered results: blue
    + one-step ahead prediction: green
    + smoothed results: red

User can change the color setting via :func:`dlm.setColor`. The color
space is the same as the `matplotlib`::

  >>> myDLM.setColor('filtered plot', 'yellow')
  >>> myDLM.setColor('data', 'blue')

If user decide to go back to the original setting, they can use
:func:`dlm.resetPlotOptions` to reset all the plot option::

  >>> myDLM.resetPlotOptions()


Advanced Settings
=================

This part of settings closely relate to the algorithm behavior and
offers some advanced features, some of which are still under
developing. Currently embedded is the :func:`dlm.stableMode` function,
which help increase the numerical stability of the :class:`dlm` when
discounting factor is used. Details about discounting factor will be
covered in next section.

In the future, following functionalities are planned to be added:
feature selection among dynamic components, factor models for high
dimensional latent states.
