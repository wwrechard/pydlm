Changelog
----------------
Updates in 0.1.1.9

* Add an example from Google data science blog and updated the homepage
* Add `dlm.plotPredictN()` which plots the prediction result from `dlm.predictN()` on top of the time series data.
* Add `dlm.predictN()` which allows prediction over multiple days.
* Change the `degree` of `trend` to match the actual meaning in polynomial, i.e, `degree=0` stands for constant and `degree=1` stands for linear trend and so on so forth.
* Add support for missing data in `modelTuner` and `.getMSE()` (Thanks @sun137653577)

Updates in 0.1.1.8

* Add an modelTuner class to auto-tune the discounting factors using gradient descent.
* Add model evaluation methods for geting residuals and MSE (one-day a head predicted loss).
* Bug fix: Fix the incorrect return length of the DLM results.
* Add travis Build test and coverage test (Thanks @liguopku).
* Modify the tests on all Numpy matrix to pass Python3 tests.
* dynamic component now accepts Numpy Matrix as feature input (Thanks @xgdgsc).
* Update the doc to be more human readable (Thanks @xgdgsc).

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
