.. py:currentmodule:: pydlm
		      
The discouting factor
=====================

The discounting factor is a technique introduced in Harrison and
West (1999) to avoid estimating the two tuning parameters in the usual
kalman filter. The basic idea is as follows.

Kalman filter assumes the same innovation for every step,
which is both hard to estimate and unnecessary. When we think of the
role of an innovation matrix, it is just to dilute the past
information and provide more uncertainty when new data comes in. Thus,
why not let the innovation depends on the posterior covariance of
the current state, for example, let the innovation be 1% of the
current posterior variance?

More precisely, suppose the prior distribution of the latent state
today is N(0, 1). After observed today's data, the posterior
distribution of the latent state becomes N(0, 0.8). Now to forecast
into tomorrow, we would like to use today's posterior to be the tomorrow's
prior distribution, with innovation added. So it is natural to relate
the innovation to the current posterior distribution. For instance, we
would like the innovation to be 1% of the current posterior, which
will be N(0, 0.08), resulting in a prior for tomorrow as N(0, 0.88).

Another way to understand the discounting factor is to think of the
information pass through. When the innovation (noise) are added, the
information of the past are diluted. Therefore, discounting factor
stands for the percentage of how much information will be passed
towards future. In the previous example, only 99% information is
passed to the new next day.

It is obvious that if 100% of information are carried over
to the future, the fitted model will be very stable, as the new data
is just a tiny portion compared to the existing information. The chain
will not moved much even an extreme point appears. By contrast, when
the discounting factor is small, say, only 90% information are passed
through every day, then any new data will account for 10% of the
performance of the time series for a given date, thus the model will
adapt to any new data or change point fairly rapid.

To set the discounting factor is simple in `pydlm`. Discounting
factors are assigned within each component, i.e., different components
can have different discounting factors::

  from pydlm import dlm, trend, seasonality
  data = [0] * 100 + [3] * 100
  myDLM = dlm(data) + trend(2, discount = 0.9) + seasonality(7,
  discount = 0.99)


Numerical stability and the renewal strategy
--------------------------------------------

One big caveat for using discounting factor is its numerical
instability. The original kalman filter already suffers from the
numerical issue due to the matrix multiplication and
subtraction. Discounting factor adds another layer to that by multiply
a scalar which is greater than 1. Usually, for models with big
discounting factor, say 0.99, its performance is close to the usual
kalman filter. However, for models with small discounting factor such
as 0.8, the numerical performance would be a nightmare. This is easy
to understand, as every step we are losing 20% information which could
means one-digit loss or so. Therefore, the fitting usually dies after around 50
steps fitting.

To amelioerate this issue, I come up with this `renewal strategy`. Notice that for
a model with a discounting factor of 0.8, fitting the whole time
series is meaningless. For data that is 40-day old, its impact on the
current state is less than 0.1%. Thus, if we disgard the whole time
series except for the last 40 days, the latent states will at most be
affected by 0.1%. This motives the following strategy: For small
discounting model, we set a acceptable threshold, say 0.1%, to compute
the corresponding renewal length. In this case, the renewal length is
about 40 days. In the actual model fitting, we then refit the model state
using only the past 40 days after every 40 steps. Following is a
simple illutration.

+---------------+---------------------------+
|day1 - day39   | regular kalman filter     |
+---------------+---------------------------+
|day40          | refit using  day1 - day39 |
+---------------+---------------------------+
|day41 - day79  | regular kalman filter     |
+---------------+---------------------------+
|day80          | refit using day41 - day79 |
+---------------+---------------------------+

With this `renewal strategy`, the longest length the discounting model
needs to fit is limited by the twice of the renewal length and thus
suppress the numerical issue.

