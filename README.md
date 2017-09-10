# Kalman Filter Controller
C++ implementation of sensor fusion on various Kalman Filter algorithms

The Kalman Filter architecture consists of two main building blocks, 
1. Individual filters
2. Fusion of filters

The fusion of filters is governed by a controller that adds multiple filters as needed. The fusion controller holds references to a multivariate gaussian mean and sigma, which the individual filters operate on. 

Currently supported filters

1. Linear - Based on Bayes filter algorithm
2. Extended - Based on Jacobian approximation for non-linear distributions
3. Unscented - Based on sigma-points as approximation of non-linear distributions

Additionally, the filters are implemented in the context of sensor fusion with different kinematic models. The table below shows the kinematic models available

| Kalman Filter      |  Radar	       |    Lidar           |
|:-------------------|:---------------:|:-------------------|
| Linear             |  -  	       |   Constant Vx, Vy  |
| Extended    	     | Constant Vx, Vy |	-           |
| Unscented	     | Constant turnrate & velocity | Constant turnrate & velocity |


The graph below shows the inheritance structure of the various filters
![alt text][image1]

## Example usage

```
FusionKF FKF(CONSTANT_VELOCITY); // Sensor fusion controller using constant Vx, Vy model
FKF.AddLaserLKF(); // adds Lidar Linear Kalman Filter to fusion controller
FKF.AddRadarEKF(); // adds Radar Extended Kalman Filter to fusion controller

FKF.ProcessMeasurements(Input) // Measurement inputs
```

```
FusionKF FKF(CONSTANT_TURNRATE_VELOCITY); // Sensor fusion controller using CTRV
FKF.AddLaserUKF(); // adds Lidar Unscented Kalman Filter to fusion controller
FKF.AddRadarUKF(); // adds Radar Unscented Kalman Filter to fusion controller

FKF.ProcessMeasurements(Input) // Measurement inputs
```

See Test directory for examples

[![Build Status](https://travis-ci.org/kernyan/KalmanFilterController.svg?branch=master)](https://travis-ci.org/kernyan/KalmanFilterController)




