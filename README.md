# Kalman Filter Controller
C++ implementation of various kalman filter types. FusionKF controller class that allows fusion of any of the selected kalman filters. 

Currently supported filters
1. Linear - Based on Bayes filter algorithm
2. Extended - Based on Jacobian approximation for non-linear distributions

Additionally, the filters are implemented in the context of sensor fusion. 

Currently supported sensor fusion
1. Radar - Using Extended KF
2. Lidar - Using Linear KF

See Test directory for examples

[![Build Status](https://travis-ci.org/kernyan/KalmanFilterController.svg?branch=master)](https://travis-ci.org/kernyan/KalmanFilterController)




