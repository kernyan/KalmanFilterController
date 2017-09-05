#include "parametric_kf.h"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;

LinearKF::LinearKF(VectorXd &Mu_in, MatrixXd &Sigma_in, long long &t_in) :
  Mu_ (Mu_in),
  Sigma_ (Sigma_in),
  previous_time_(t_in)
{
}

void LinearKF::Initialize(MatrixXd &TransFunc_in, 
  MatrixXd &TransSigma_in,
  MatrixXd &MeasFunc_in,
  MatrixXd &MeasSigma_in){

  At_ = TransFunc_in;
  Rt_ = TransSigma_in;
  Ct_ = MeasFunc_in;
  Qt_ = MeasSigma_in;
}

void LinearKF::CalculateMuBar(){
  MuBar_ = At_ * Mu_;
}

void LinearKF::CalculateSigmaBar(){
  SigmaBar_ = At_ * Sigma_ * At_.transpose() + Rt_;
}

MatrixXd LinearKF::CalculateMeasurementVar(){
  return Ct_ * SigmaBar_ * Ct_.transpose() + Qt_;
}

VectorXd LinearKF::CalculatePredictedMeasurement(){
  return Ct_ * MuBar_;
}

void LinearKF::FirstTimeStep(MeasurementPackage &meas_in){
  previous_time_ = meas_in.timestamp_;
  Mu_ = Ct_.transpose() * meas_in.raw_measurements_;
}

void LinearKF::Step(MeasurementPackage &meas_in){
  CalculateMuBar();
  CalculateSigmaBar();
  MatrixXd S = CalculateMeasurementVar();
  MatrixXd K = SigmaBar_ * Ct_.transpose() *  S.inverse();
  VectorXd zhat = CalculatePredictedMeasurement();
  Mu_ = MuBar_ + K * (meas_in.raw_measurements_ - zhat);
  Sigma_ = SigmaBar_ - K * S * K.transpose();
}


LaserLKF::LaserLKF(VectorXd &Mu_in, MatrixXd &Sigma_in, long long &t_in) :
  LinearKF(Mu_in, Sigma_in, t_in),
  noise_ax_(-1),
  noise_ay_(-1)
{
}

void LaserLKF::Initialize(){

  noise_ax_ = 9;
  noise_ay_ = 9;

  MatrixXd At(4,4);
  At << 1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1;

  MatrixXd Rt(4,4);

  MatrixXd Ct(2,4);
  Ct << 1,0,0,0,
        0,1,0,0;

  MatrixXd Qt(2,2);
  Qt << 0.0225,     0,
             0,0.0225;

  LinearKF::Initialize(At, Rt, Ct, Qt);
}

void LaserLKF::Step(MeasurementPackage &meas_in){

  if (Sensor_ != meas_in.sensor_type_) return;

  double dt = meas_in.timestamp_ - previous_time_;
  previous_time_ = meas_in.timestamp_;

  dt *= 0.000001; // scale to seconds
  At_(0,2) = dt;
  At_(1,3) = dt;

  float dt_2 = dt*dt;
  float dt_3 = dt_2*dt;
  float dt_4 = dt_3*dt;
  Rt_ << dt_4/4*noise_ax_, 0, dt_3/2*noise_ax_, 0,
         0, dt_4/4*noise_ay_, 0, dt_3/2*noise_ay_,
         dt_3/2*noise_ax_, 0, dt_2*noise_ax_, 0,
         0, dt_3/2*noise_ay_, 0, dt_2*noise_ay_;
  
  LinearKF::Step(meas_in);
}


ExtendedKF::ExtendedKF(VectorXd &Mu_in, MatrixXd &Sigma_in, long long &t_in) :
    Mu_ (Mu_in),
    Sigma_ (Sigma_in),
    previous_time_ (t_in)
{
}
  
void ExtendedKF::Initialize(Func &g_in,
    MatrixXd &G_in,
    MatrixXd &Rt_in,
    Func &h_in,
    function<MatrixXd (VectorXd)> &H_in,
    MatrixXd &Qt_in){
  g_ = g_in;
  G_ = G_in;
  Rt_ = Rt_in;
  h_ = h_in;
  H_ = H_in;
  Qt_ = Qt_in;
}

void ExtendedKF::CalculateMuBar(){
  MuBar_ = g_(Mu_);
}

void ExtendedKF::CalculateSigmaBar(){
  SigmaBar_ = G_ * Sigma_ * G_.transpose() + Rt_;
}

MatrixXd ExtendedKF::CalculateMeasurementVar(){
  MatrixXd Ht = H_(MuBar_);
  return Ht * SigmaBar_ * Ht.transpose() + Qt_;
}

VectorXd ExtendedKF::CalculatePredictedMeasurement(){
  return h_(MuBar_);
}

void ExtendedKF::Step(MeasurementPackage &meas_in){
  CalculateMuBar();
  CalculateSigmaBar();
  MatrixXd S = CalculateMeasurementVar();
  MatrixXd Ht = H_(MuBar_);
  MatrixXd K = SigmaBar_ * Ht.transpose() * S.inverse();
  VectorXd zhat = CalculatePredictedMeasurement();
  VectorXd y = meas_in.raw_measurements_ - zhat;
  if (meas_in.raw_measurements_(1) * zhat(1) <= 0){
    y << 0,0,0; // do not update when theta flips sign
  }
  Mu_ = MuBar_ + K * y;
  Sigma_ = SigmaBar_ - K * S * K.transpose();
}


RadarEKF::RadarEKF(VectorXd &Mu_in, MatrixXd &Sigma_in, long long &t_in) :
  ExtendedKF(Mu_in, Sigma_in, t_in),
  noise_ax_ (-1),
  noise_ay_ (-1)
{
}


void RadarEKF::Step(MeasurementPackage &meas_in){

  if (Sensor_ != meas_in.sensor_type_) return;

  double dt = meas_in.timestamp_ - previous_time_;
  previous_time_ = meas_in.timestamp_;

  dt *= 0.000001; // scale to seconds
  G_(0,2) = dt;
  G_(1,3) = dt;

  auto G_in = G_;
  g_ = [G_in](VectorXd Mu_in){
    return G_in*Mu_in;
  };

  float dt_2 = dt*dt;
  float dt_3 = dt_2*dt;
  float dt_4 = dt_3*dt;
  Rt_ << dt_4/4*noise_ax_, 0, dt_3/2*noise_ax_, 0,
         0, dt_4/4*noise_ay_, 0, dt_3/2*noise_ay_,
         dt_3/2*noise_ax_, 0, dt_2*noise_ax_, 0,
         0, dt_3/2*noise_ay_, 0, dt_2*noise_ay_;
  
  ExtendedKF::Step(meas_in);
}


void RadarEKF::Initialize(){

  noise_ax_ = 9;
  noise_ay_ = 9;

  MatrixXd Gt(4,4);
  Gt << 1,0,0,0,
        0,1,0,0,
        0,0,1,0,
        0,0,0,1;

  function<VectorXd (VectorXd)> g_in; 
  MatrixXd Rt(4,4);
  
  function<VectorXd (VectorXd)> h_in = [](VectorXd MuBar_in){
    VectorXd zhat(3);

    float px = MuBar_in(0);
    float py = MuBar_in(1);
    float vx = MuBar_in(2);
    float vy = MuBar_in(3);
    
    float p = sqrt(px*px + py*py);
    float phi = atan2(py, px);
    if (phi > -M_PI){
      while (phi > M_PI) phi -= 2*M_PI;
    } else {
      while (phi < -M_PI) phi += 2*M_PI;
    }

    float p_dot = (px*vx+py*vy)/p;
    zhat << p, phi, p_dot;
    return zhat;
  };

  function<MatrixXd (VectorXd)> H_in = [](VectorXd MuBar_in){
    MatrixXd Hj(3,4);

    float px = MuBar_in(0);
    float py = MuBar_in(1);
    float vx = MuBar_in(2);
    float vy = MuBar_in(3);

    float p_mag = px*px + py*py;
    float p_dot = sqrt(p_mag);

    if (p_mag < 0.001) return Hj;

    Hj << px/p_dot, py/p_dot, 0, 0,
         -py/p_mag, px/p_mag, 0, 0,
         py*(vx*py-vy*px)/(p_mag*p_dot), px*(vy*px-vx*py)/(p_mag*p_dot), px/p_dot, py/p_dot;
    
    return Hj;
  };

  MatrixXd Qt(3,3);
  Qt << 0.09,0     ,0,
        0   ,0.0009,0,
        0   ,0     ,0.09;

  ExtendedKF::Initialize(g_in, Gt, Rt, h_in, H_in, Qt);
}
