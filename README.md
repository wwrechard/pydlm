[PyDLM-lite](https://pydlm.github.io/)  [![Build Status](https://travis-ci.org/wwrechard/pydlm.svg?branch=master)](https://travis-ci.org/wwrechard/pydlm) [![Coverage Status](https://coveralls.io/repos/github/wwrechard/pydlm/badge.svg?branch=master)](https://coveralls.io/github/wwrechard/pydlm?branch=master)
=======================================================


The lite version of the main `pydlm` package where the plotting functionality and the dependency on the `matplotlib` has been removed. Most refactoring work will be conducted on this package to improve the class on multi-threading and online learning. In the meanwhile, the main `pydlm` package will remain in its current structure for future development.

Going forward, the two packages will be developed under two different principles:

1. `pydlm` will support more sophisticated models and more advanced algorithm such as sequential monte carlo. The algorithm will be optimized in terms of accuracy rather than latency. The primary use case is on advanced inference and data analysis with small datasets.

2. `pydlm-lite` will mainly focus on normal-normal and poisson-gamma models with the fastest possible fitting algorithm. The class design will support concurrency and online updating. The primary use case is scalable anomaly detection and forecasting with millions of time series.

Updates in the github version
-------------------------------------------
* Plan to refactor the `dlm` class. Align with the goal to separate model and data, I'm going to refactor the `dlm` class such that
  1. The main `dlm` class will only contain model build information and is supposed to be 'const' after construction.
  2. Time series data will be passed in as an argument to the `fit` or `forwardFilter` and the fitted result will be returned as well as the model status.
  3. Model status can also be passed into `fit` and `forwardFilter` as a prior.
  The goal is to make the `dlm` class state-independent, so that the class is thread-safe and can be shared by multiple threads for parallel processing. While in progress, all the old class behavior will be kept.

Documentation
-------------
Detailed documentation is provided in [PyDLM](https://pydlm.github.io/) with special attention to the [User manual](https://pydlm.github.io/#dynamic-linear-models-user-manual).
