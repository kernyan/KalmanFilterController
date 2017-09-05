#include "fusion_kf.h"
#include "Eigen/Dense"
#include <iostream>
#include <vector>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

FusionKF::FusionKF() :
  IsFirstTime (true),
  previous_time_(-1)
{
  Mu_ = VectorXd(4);
  Mu_ << 0,0,0,0;

  Sigma_ = MatrixXd(4,4);
  Sigma_ << 1,0,0,0,
            0,1,0,0,
            0,0,1000,0,
            0,0,0,1000;
}

FusionKF::~FusionKF(){
  for (auto &each : filters_){
    delete each;
    each = nullptr;
  }
}

void FusionKF::AddLaserLKF(){
  
  // Laser Filter
  
  LaserLKF* LaserFilter = new LaserLKF(Mu_, Sigma_, previous_time_);
  LaserFilter->Initialize();
  filters_.push_back(LaserFilter);
}

void FusionKF::AddRadarEKF(){
  
  // Radar Filter
  
  RadarEKF *RadarFilter = new RadarEKF(Mu_, Sigma_, previous_time_);
  RadarFilter->Initialize();
  filters_.push_back(RadarFilter);
}

void FusionKF::ProcessMeasurement(MeasurementPackage &meas_in){
  if (IsFirstTime){
    Mu_(0) = meas_in.raw_measurements_(0);
    Mu_(1) = meas_in.raw_measurements_(1);
    assert(previous_time_ == -1);
    previous_time_ = meas_in.timestamp_;
    
    IsFirstTime = false;
  }

  for (auto &filter : filters_){
    filter->Step(meas_in);
  }
}


VectorXd CalculateRMSE(const vector<VectorXd> &estimations,
                       const vector<VectorXd> &ground_truth){
  VectorXd rmse(4);
  rmse << 0,0,0,0;

  if (estimations.size() != ground_truth.size()
     || estimations.size() == 0){
    cout << "Invalud estimation or ground truth data" << endl;
    return rmse;
  }

  for (int i = 0; i < estimations.size(); ++i){
    VectorXd residual = estimations[i] - ground_truth[i];
    residual = residual.array()*residual.array();
    rmse += residual;
  }

  rmse = rmse/estimations.size();
  rmse = rmse.array().sqrt();
  return rmse;
}

