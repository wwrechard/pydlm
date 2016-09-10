.. currentmodule:: pydlm

This package implements the Bayesian dynamic linear model (DLM, Hurrison
and West, 1999) for time series analysis.
The DLM is built upon two layers. The first layer is
the fitting algorithm. DLM adopts a modified Kalman filter with a
unique discounting technique from
Hurrison and West (1999). Like the usual Kalman filter, it accepts a
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
Hurrison and West (1999), most common models can be expressed in
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


Basic functionality
===================

