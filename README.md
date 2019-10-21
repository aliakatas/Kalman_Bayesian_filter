# Hybrid Kalman-Bayesian filter
Hybrid filter for post-processing forecasts provided by Numerical Weather/Wave Prediction (NWP) models.
This work is based on this publication: [A hybrid Bayesian Kalman filter and applications to numerical wind speed
modeling](http://dx.doi.org/10.1016/j.jweia.2017.04.007)

It implements each module of the combined system, as well as the integrated version of the full Hybrid filter.
The aim of this tool is to remove systematic and random errors from model predictions as much as possible, leading to more accurate forecasts.

Currently, it is able to handle data following Normal, Log-normal and Weibull (min) distributions. The error of the model predictions is always assumed to follow Normal distribution.



