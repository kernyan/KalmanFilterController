#ifndef FusionEKF_H_
#define FusionEKF_H_

#include "measurement_package.h"
#include "Eigen/Dense"
#include <vector>
#include <list>
#include <string>
#include <fstream>
#include "parametric_kf.h"

using namespace std;

class FusionKF {
public:
  
  list<ParametricKF*> filters_;
  void AddLaserLKF();
  void AddRadarEKF();
  void ProcessMeasurement(MeasurementPackage &meas_in);
  VectorXd GetMu() const {return Mu_;};
  MatrixXd GetSigma() const {return Sigma_;};

  FusionKF();
  ~FusionKF();

private:

  VectorXd Mu_;
  MatrixXd Sigma_;
  long long previous_time_;
  bool IsFirstTime;
};

VectorXd CalculateRMSE(const vector<VectorXd> &estimations,
                       const vector<VectorXd> &ground_truth);
#endif /* FUSION_EKF_H */