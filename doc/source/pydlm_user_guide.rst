.. py:currentmodule:: pydlm
		      
Dynamic linear models --- user manual
=====================================

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

  myDLM = dlm(data) + trend(1)

The modeling process is simple.

The purpose of the modeling is to better understand the time series
data and for to forecast into the future. So the key output from the
model are the filtered time series, smoothed time series and one-step
ahead prediction. We will cover this topic later in this section.

The advantage of :mod:`pydlm`:

    + flexibility in constructing complicated models

    + Extremely efficient model fitting with the discounting technique

    + user-specific adjustment on the adaptive property of the model

The disadvantage of :mod:`pydlm`:

    + only for Gaussian noise


Miscellaneous
-------------
+ `PyDLM` index starts at 0 instead of 1, i.e., for any prediction or
  modification that involves `date` argument, it corresponds to the
  actual index in the array. For instance, for a time series with
  length 10, the date of the last day is 9. 

Modeling
--------

As discussed in the beginning, the modeling process is very simple
with :mod:`pydlm`, most modeling functions are integrated in the class
:class:`dlm`. Following is an example for constructing a dlm with
linear trend, 7-day seasonality and control variables::

  from pydlm import dlm, trend, seasonality, dynamic, autoReg, longSeason
  data = [0] * 100 + [3] * 100
  SP500Index = [[2000] for i in range(100)] + [[2010] for i in range(100)]
  page = [[i, i + 1, i + 2, i + 3] for i in range(200)]
  myDLM = dlm(data)
  myDLM = myDLM + trend(degree=1, discount=0.95, name='trend1')
  myDLM = myDLM + seasonality(period=7, discount=0.99, name='week')
  myDLM = myDLM + dynamic(features=SP500Index, discount=1, name='SP500')
  myDLM = myDLM + dynamic(features=page, discount=1, name='page')

User can also use :func:`dlm.add` method to add new component::

  myDLM.add(trend(degree=0, discount=0.99, name='trend2'))

The imput :attr:`data` must be an 1d array or a list, since the current
:class:`dlm` only supports one dimensional time series. Supporting for
multivariate time series will be built upon this one dimensional class
and added in the future.

Now the variable `myDLM` contains the data and the modeling
information. It will construct the corresponding transition,
measurement, innovation, latent states and error covariance matrix
once model fitting is called. Modify an existing model is also
simple. User can brows the existing components of the model by::

  myDLM.ls()

It will show all the existing components and their corresponding
names. Name can be specified when the component is added to the `dlm`,
for example::

  myDLM = myDLM + seasonality(4, name = 'day4')
  myDLM.ls()

We can also easily delete the unwanted component by using `delete`::

  myDLM.delete('trend2')
  myDLM.delete('day4')

After the building steps, we can specify some parameters for the model
fitting, the most common one is the prior guess on the observational
noise. The default value is 1.0. To change that to 10 you can do::

  myDLM.noisePrior(10.0)

Such change usually has small impact on the model and is almost
ignorable.

Model components
----------------

There are four model components provided with this
package: trend, seasonality, dynamic and the auto-regression.

Trend
`````
:class:`trend` class is a model component for trend
behavior. The data might be const or increasing linearly or
quadraticly, which can all be captured by :class:`trend`. The degree
argument specifics the shape of the trend. `degree=0` indicates this
is a const, `degree=1` indicates a line and `degree=2` stands for a
quadratic curve and so on so forth. `w` sets the prior
covariance for the trend component (same for all the other
components). The discounting factor will be explained later in next
section::

  linearTrend = trend(degree=1, discount=0.99, name='trend1', w=1e7)

Seasonality
```````````
The :class:`seasonality` class models the periodic behavior of the
data. Compared to the sine or cosine periodic curves,
:class:`seasonality` in this packages is more flexible, since it can
turn into any shapes, much broader than the triangular families::

  weekPeriod = seasonality(period=7, discount=0.99, name='week', w=1e7)

In the package, we implements the seasonality component in a
`form-free` way (Harrison and West, 1999) to avoid the identifiability
issue. The states of one seasonality component are always summed up to
zero, so that it will not tangle with the :class:`trend` component.

Dynamic
```````
The :class:`dynamic` class offers the modeling ability to add any additional
observed time series as a controlled variable to the current one. For
example, when studying the stock price, the 'SP500' index could be a
good indicator for the modeling stock. A dynamic component need the
user to supply the necessary information of the control variable over
time::

  SP500 = dynamic(features=SP500Index, discount=0.99, name='SP500', w=1e7)

The input :attr:`features` for :class:`dynamic` should be a list of
lists, since multi-dimension features are allowed. Following is one
simple example::

  Features = [[2000], [2010], [2020], [2030]]
  Features = [[1.0, 2.0], [1.0, 3.0], [3.0, 3.0]]


Auto-regression
```````````````
The :class:`autoReg` class constructs the auto-regressive component on
the model, i.e., the direct linear or non-linear dependency between
the current observation and the previous days. User needs to specify
the number of days of the dependency::

  AR3 = autoReg(degree=3, discount=0.99, name='ar3', w=1e7)

There is once a `data` argument needed for constructing autoregression
features but is now deprecated. :class:`autoReg` can now fetch the
data directly from the main :class:`dlm` class and no need to provide
during instantiation.

In this example, the latent stats for Auto-regression are aligned in a
way of [today - 3, today - 2, today - 1]. So when fetching the
coefficients from the latent states, this will be the correct order to
read the coefficients.

Long-seasonality
````````````````
The :class:`longSeason` class is a complement class for
:class:`seasonality`. It allows constructing seasonality component that
does not change every step. For example, the time unit is day, but
user wants to add a monthly seasonality, then :class:`longSeason` is
the correct choice::

  monthly = longSeason(period=12, stay=30, data=data, name='monthly', w=1e7)

These five classes of model components offer abundant modeling
possiblities of the Bayesian dynamic linear model. Users can construct
very complicated models using these components, such as hourly, weekly or
monthly periodicy and holiday indicator and many other features.

Model fitting
-------------

Entailed before, the fitting of the dlm is fulfilled by a modified
Kalman filter. Once the user finished constructing the model by adding
different components. the :class:`dlm` will compute all the necessary
quantities internally for using Kalman filter. So users can simply
call :func:`dlm.fitForwardFilter`, :func:`dlm.fitBackwardSmoother` or
even simply :func:`dlm.fit` to fit both forward filter and backward
smoother::

  myDLM.fitForwardFilter()
  myDLM.fitBackwardSmoother()
  myDLM.fit()

The :func:`dlm.fitForwardFilter` is implemented in an online
manner. It keeps an internal count on the filtered dates and once new
data comes in, it only filters the new data without touching the
existing results. In addition, this function also allows a rolling
window fitting on the data, i.e., there will be a moving window and
for each date, the kalman filter will only use the data within the
window to filter the observation. This is equivalent to that the model
only remembers a fixed length of dates::

  myDLM.fitForwardFilter(useRollingWindow=True, windowLength=30)
  myDLM.fitBackwardSmoother()

For :func:`dlm.backwardSmoother`, it has to use the whole time series
to smooth the latent states once new data comes in. The smoothing
provides a good retrospective analysis on our past decision of the
data. For example, we might initially believe the time series is
stable, while that could be a random behavior within a volatile time
series, and the user learn this from the smoother.

Once the model fitting is completed, users can fetch the filtered or
smoothed results from :class:`dlm`::

  myDLM.getMean(filterType='forwardFilter')
  myDLM.getMean(filterType='backwardSmoother')
  myDLM.getMean(filterType='predict')

  myDLM.getVar(filterType='forwardFilter')
  myDLM.getVar(filterType='backwardSmoother')
  myDLM.getVar(filterType='predict')

The :class:`dlm` recomputes a wide variety of model quantities that
can be extracted by the user. For example, user can get the filtered
states and covariance by typing::

  myDLM.getLatentState(filterType='forwardFilter')
  myDLM.getLatentState(filterType='backwardSmoother')

  myDLM.getLatentCov(filterType='forwardFilter')
  myDLM.getLatentCov(filterType='backwardSmoother')

This can be specified into individual component. For example, assume the
model contains a :class:`trend` component with a name of `trend1`, we
can extract the corresponding latent state only for `trend1` as::

  myDLM.getLatentState(filterType='forwardFilter', name='trend1')
  myDLM.getLatentState(filterType='backwardSmoother', name='trend1')

  myDLM.getLatentCov(filterType='forwardFilter', name='trend1')
  myDLM.getLatentCov(filterType='backwardSmoother', name='trend1')

as well as the mean of `trend1` (evaluation * latent states)::

  myDLM.getMean(filterType='forwardFilter', name='trend1')
  myDLM.getVar(filterType='forwardFilter', name='trend1')

One can also get the confidence interval on the filtered time series::

  myDLM.geInterval(filterType='forwardFilter', p = 0.99)

There are also corresponding methods for smoothed and predicted
results. For more detail, please refer to the :class:`dlm` class
documentation.

Model prediction
----------------
:class:`dlm` provides three predict functions: :func:`dlm.predict` and
:func:`dlm.continuePredict` and :func:`dlm.predictN`. The last one is
wrapper of the former two and is recommended to use. (The former two
will be gradually deprecated).

The :func:`dlm.predict` is a one-day ahead
prediction function based on a user given date and feature set::

  # predict next date after the time series
  featureDict = {'SP500':[2090], 'page':[1, 2, 3, 4]}
  (predictMean, predictVar) = myDLM.predict(date=myDLM.n - 1, featureDict=featureDict)

The function returns a tuple of predicted mean and predicted variance.
The `featureDict` argument is a dictionary contains the feature
information for :class:`dynamic` component. Suppose the model contains
a one-dimensional dynamic component named `SP500` and another
four-dimensional dynamic component `page`, then the featureDict takes the
following Form::

  featureDict = {'SP500':[2090], 'page':[1, 2, 3, 4]}

If the `featureDict` is not supplied but the date is not the last day,
then the algorithm will automatically fetch from the old data about
the feature value of all the dynamic component::

  # predict a day in the middle
  (predictMean, predictVar) = myDLM.predict(date=myDLM.n - 10)

The algorithm will use the feature on the date of `myDLM.n - 9` in
`featureDict`. If date is the last day but the featureDict is not
provided, then an error will be raised.

If the user is interested beyond one-day ahead prediction, they can
use :func:`dlm.continuePredict` for multiple-day ahead prediction,
after using :func:`dlm.predict`::

  feature1 = {'SP500':[2090], 'page':[10, 20, 30, 40]}
  feature2 = {'SP500':[2010], 'page':[11, 21, 31, 41]}
  feature3 = {'SP500':[1990], 'page':[12, 22, 32, 42]}

  # one-day ahead prediction after the last day
  (predictMean, predictVar) = myDLM.predict(date=myDLM.n - 1, featureDict=feature1)
  # we continue to two-day ahead prediction after the last day
  (predictMean, predictVar) = myDLM.continuePredict(featureDict=feature2)
  # we continue to three-day ahead prediction after the last day
  (predictMean, predictVar) = myDLM.continuePredict(featureDict=feature3)

:func:`dlm.continuePredict` can only be used after :func:`dlm.predict`
for multiple-day prediction. The `featureDict` can also be ignored if
the prediction is requested on dates before the last day and the
features on the predict day can be found from the old data.

:func:`dlm.predictN` (recommended) predicts over multiple days and is
a wrapper of the two functions above. Using the same example, the
results can be obtained by just called::

  features = {'SP500':[[2090], [2010], [1990]], 'page':[[10, 20, 30,
  40], [11, 21, 31, 41], [12, 22, 32, 42]]}
  (predictMean, predictVar) = myDLM.predictN(N=3, date=myDLM.n-1,
  featureDict=features)

The `predictMean` and `predictVar` will be two lists of three elements
containing the predicted mean and variance.
      
Model amending
--------------

The user can still add, delete, modify data even when the model has
been constructed.

Adding new data
```````````````
For adding more data, user can opt to
:func:`dlm.append`::

   newData = [0, 1, 2]
   myDLM.append(newData, component = 'main')

If the model contains :class:`dynamic` component, the corresponding
features need to be updated as well::

  newSP500 = [[2000], [2100], [2200]]
  myDLM.append(data = newSP500, component = 'SP500')

Then the user can rerun the forward filter::

  myDLM.fitForwardFilter()

The package will continue running the forward filter on the three new
data ponts.

Deleting existing data
``````````````````````
To delete any existing data, user can simply use the :func:`dlm.popout`
function from :class:`dlm` on a specific date, for example::

  myDLM.popout(0)

Different from :func:`dlm.append`, :func:`dlm.popout` will be executed
automatically for all components, so the user does not need to conduct
the deletion mannually for each component. After the deletion, the
forward filter needs to be rerun following the deleted date::

  myDLM.fitForwardFilter()

Again, the package will automatically recognize the date and fit only
the necessary period of time.

Ignoring a date
```````````````
Ignoring is very similar to deleting. The only difference is the time
counts. Because deleting will delete the data entirely, the time
counts will therefore reduce by 1. By contrast, ignoring will treat
the specific date as missing data, so the time count will not
change. This difference is important when preriodicy is
concerned. Changing of time counts will have high impacts on
:class:`seasonality` components.

:func:`dlm.ignore` simply set the data of a specific date to be None::

  myDLM.ignore(2)

modify data
```````````
The :class:`dlm` also provides user the ability to modify the data on a
specific date and a specific component. This function enables possible
future extension to interactive anomaly detection and data debugging::

  myDLM.alter(date = 2, data = 0, component = 'main')


Model plotting
--------------

This package offers rich ploting options for illustrating the
results. User can simply call :func:`dlm.plot` to directly plot the
results once the models are fitted::

  myDLM.plot()

  # plot the mean of a given component
  myDLM.plot(name=the_component_name)

  # plot the latent state of a given component
  myDLM.plotCoef(name=the_component_name)

User can choose which results to plot via :func:`dlm.turnOn` and
:func:`dlm.turnOff`::

   myDLM.turnOn('filtered plot')
   myDLM.turnOff('predict plot')
   myDLM.turnOff('smoothed plot')

User can also choose whether to plot the confidence interval and
whether plot the results in one figure or separate figures. The
default is to plot the confidence interval and in separate plots. To
change that::

  myDLM.turnOff('confidence')
  myDLM.turnOff('multiple plots')

The quantile of the confidence interval can be set via
:func:`dlm.setConfidence`::

  myDLM.setConfidence(p = 0.95)

Currently there are two types of confidence interval realization.
The default is 'ribbon' and the alternative is 'line'. Users can
change the confidence interval plots by::

  myDLM.setIntervalType('ribbon')
  myDLM.setIntervalType('line')

The default colors for the plots are:

    + original data: 'black'
    + filtered results: 'blue'
    + one-step ahead prediction: 'green'
    + smoothed results: 'red'

User can change the color setting via :func:`dlm.setColor`. The color
space is the same as the `matplotlib`::

  myDLM.setColor('filtered plot', 'yellow')
  myDLM.setColor('data', 'blue')

If user decide to go back to the original setting, they can use
:func:`dlm.resetPlotOptions` to reset all the plot option::

  myDLM.resetPlotOptions()

Model Tuning and evaluation
---------------------------
The discounting factor of DLM determines how fast the model adapts to
the new data. It could cause troublesome when users actually want the
model itself to figure that out. The :class:`modelTuner` provides
users automatic tool for tuning the discounting factors:: 

  from pydlm import modelTuner
  myTuner = modelTuner(method='gradient_descent', loss='mse')
  tunedDLM = myTuner.tune(myDLM, maxit=100)

The tuned discounting factor will be set inside the `tunedDLM`. Users
can examine the tuned value from `myTuner`::

  tuned_discounts = myTuner.getDiscounts()

After tuning, `myDLM` will remain unchanged and `tunedDLM` will
contain the tuned discounting factos and will be in uninitialized
status. Users need to run::

  tunedDLM.fit()

before any other analysis on `tunedDLM`. The :class:`dlm` also provides
a simpler way to tune the discounting factor if the user would like
`myDLM` to be altered directly::

  myDLM.tune()
  
The tuner makes use of the MSE (one-day ahead prediction loss) and the
gradient descent algorithm to tune the discounting factor to achieve
the minimum loss. The discounting factors are assumed to be different
across components but the same within a component. For now, only the
MSE loss and the gradient descent algorithm is supported.

To ease the evaluation of the performance of the model fitting, the
model also provides the residual time series and the one-day a head
prediction error via::

  mse = myDLM.getMSE()
  residual = myDLM.getResidual(filterType='backwardSmoother')

Users can use these values to evaluate and choose the optimal model.

Advanced Settings
-----------------

This part of settings closely relate to the algorithm behavior and
offers some advanced features, some of which are still under
development. Currently implemented is the :func:`dlm.stableMode`
function and the :func:`dlm.evolveMode`. The :func:`dlm.stableMode` is
turned on by default, and you can turn it off by::

  myDLM.stableModel(False)

This mode helps increasing the numerical stability of the :class:`dlm`
when small discounting factor is used. Details about discounting
factor will be covered in next section. The :func:`dlm.evolveMode` is
used to control how different components evolve over time. See
Harrison and West (1999, Page 202). They could
evolve independently, which is equivalent to assume the innovation
matrix is a block-diagonal matrix. The default assumption is
'independent' and to change to 'dependent', we can simply type::

  myDLM.evolveMode('dependent')

The difference between 'independent' and 'dependent' is best explained
when there are multiple components with different discounting factor
and one of them is One. In the 'dependent' mode, the smoothed latent
states of the component with discount factor 1 will be a value
fluctuating around a constant, while in the 'independent' mode, it
will be an exact constant. User can choose which to use depending on
their own use case.

In the future, following functionalities are planned to be added:
feature selection among dynamic components, factor models for high
dimensional latent states.
