
#include "fusion_kf.h"
#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "measurement_package.h"
#include <iostream>
#include <vector>

using namespace std;

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

class LKFTest : public :: testing::Test {
  protected:

  virtual void SetUp(){

    mu_ = VectorXd(4);
    mu_ << 0.312242687, 0.580339789,1,1;

    Sigma_ = MatrixXd(4,4);
    Sigma_ << 1,0,0,0,
              0,1,0,0,
              0,0,1000,0,
              0,0,0,1000;
    t_ = 0;

    LaserLKF *LaserFilter = new LaserLKF(mu_, Sigma_, t_);
    LaserFilter->Initialize();
    ParaKF_ = LaserFilter;

    meas_.sensor_type_ = LASER;
    meas_.raw_measurements_ = VectorXd(2);
    meas_.raw_measurements_ << 0.312242687, 0.580339789;
    meas_.timestamp_ = t_ + 100000;
  }

  virtual void TearDown() {
    delete ParaKF_;
    ParaKF_ = nullptr;
  }

  VectorXd mu_;
  MatrixXd Sigma_;
  long long t_;
  ParametricKF *ParaKF_;
  MeasurementPackage meas_;
};

TEST_F(LKFTest, PredictionMu){
  ParaKF_->Step(meas_);
  auto LKF = dynamic_cast<LinearKF*> (ParaKF_);
  auto Mat = LKF->GetMuBar();
  vector<float> Calc(Mat.data(),Mat.data()+Mat.rows()*Mat.cols());
  vector<float> Ans {0.412242687, 0.680339789,1,1};

  for (int i = 0; i < Calc.size(); ++i){
    EXPECT_NEAR(Calc[i], Ans[i], 0.00001) << "for i: " << i;
  }
}


TEST_F(LKFTest, PredictionSigma){
  ParaKF_->Step(meas_);
  auto LKF = dynamic_cast<LinearKF*> (ParaKF_);
  auto Mat = LKF->GetSigmaBar();
  vector<float> Calc(Mat.data(),Mat.data()+Mat.rows()*Mat.cols());
  vector<float> Ans {11.0002, 0, 100.005, 0,
                          0, 11.0002, 0, 100.005,
                          100.005, 0, 1000.09, 0,
                          0, 100.005, 0, 1000.09};

  for (int i = 0; i < Calc.size(); ++i){
    EXPECT_NEAR(Calc[i], Ans[i], 0.001) << "for i: " << i;
  }
}


TEST_F(LKFTest, PredictAndUpdate){
  ParaKF_->Step(meas_);
  meas_.raw_measurements_ << 0.6, 0.6;
  meas_.timestamp_ += 100000;
  ParaKF_->Step(meas_);
  auto LKF = dynamic_cast<LinearKF*> (ParaKF_);
  vector<float> Calc(mu_.data(),mu_.data()+mu_.rows()*mu_.cols());
  vector<float> Ans {0.593825,0.599774,2.69674,0.188019};

  vector<float> Calc2(Sigma_.data(), Sigma_.data() + Sigma_.rows() * Sigma_.cols());
  vector<float> Ans2 {0.0220007,0,0.210544,0,
                           0,0.0220007,0,0.210544,
                           0.210544,0,4.09938,0,
                           0,0.210544,0,4.09938};

  for (int i = 0; i < Calc.size(); ++i){
    EXPECT_NEAR(Calc[i], Ans[i], 0.001) << "for i: " << i;
  }

  for (int i = 0; i < Calc2.size(); ++i){
    EXPECT_NEAR(Calc2[i], Ans2[i], 0.001) << "for i: " << i;
  }
}


class RKFTest : public :: testing::Test {
  protected:

  virtual void SetUp(){

    mu_ = VectorXd(4);
    mu_ << 1,1,1,1;

    Sigma_ = MatrixXd(4,4);
    Sigma_ << 1,0,0,0,
              0,1,0,0,
              0,0,1000,0,
              0,0,0,1000;

    t_ = 0;

    ExtendedKF *EFilter = new ExtendedKF(mu_, Sigma_, t_);

    MatrixXd Gt(4,4);
    float dt = 0.1;
    Gt << 1,0,dt, 0,
          0,1, 0,dt,
          0,0, 1, 0,
          0,0, 0, 1;

    function<VectorXd (VectorXd)> g_in = [Gt](VectorXd Mu_in){
      return Gt*Mu_in;
    };

    MatrixXd Rt(4,4);
    float noise_ax = 9;
    float noise_ay = 9;
    float dt_2 = dt*dt;
    float dt_3 = dt_2*dt;
    float dt_4 = dt_3*dt;
    Rt << dt_4/4*noise_ax, 0, dt_3/2*noise_ax, 0,
          0, dt_4/4*noise_ay, 0, dt_3/2*noise_ay,
          dt_3/2*noise_ax, 0, dt_2*noise_ax, 0,
          0, dt_3/2*noise_ay, 0, dt_2*noise_ay;

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

    EFilter->Initialize(g_in, Gt, Rt, h_in, H_in, Qt);
    ParaKF_ = EFilter;

    meas_.sensor_type_ = RADAR;
    meas_.raw_measurements_ = VectorXd(3);
    meas_.raw_measurements_ << 1.01489198, 0.554329216,4.89280701; 
    meas_.timestamp_ = t_ + 100000;
  }

  virtual void TearDown() {
    delete ParaKF_;
    ParaKF_ = nullptr;
  }

  VectorXd mu_;
  MatrixXd Sigma_;
  long long t_;
  ParametricKF *ParaKF_;
  MeasurementPackage meas_;
};

TEST_F(RKFTest, PredictionMu){
  ParaKF_->Step(meas_);
  auto EKF = dynamic_cast<ExtendedKF*> (ParaKF_);
  auto Mat = EKF->GetMuBar();
  vector<float> Calc(Mat.data(),Mat.data()+Mat.rows()*Mat.cols());
  vector<float> Ans {1.1,1.1,1,1};

  for (int i = 0; i < Calc.size(); ++i){
    EXPECT_NEAR(Calc[i], Ans[i], 0.00001) << "for i: " << i;
  }
}

TEST_F(RKFTest, PredictionSigma){
  ParaKF_->Step(meas_);
  auto RKF = dynamic_cast<ExtendedKF*> (ParaKF_);
  auto Mat = RKF->GetSigmaBar();
  vector<float> Calc(Mat.data(),Mat.data()+Mat.rows()*Mat.cols());
  vector<float> Ans {11.0002, 0, 100.005, 0,
                     0, 11.0002, 0, 100.005,
                     100.005, 0, 1000.09, 0,
                     0, 100.005, 0, 1000.09};

  for (int i = 0; i < Calc.size(); ++i){
    EXPECT_NEAR(Calc[i], Ans[i], 0.001) << "for i: " << i;
  }
}

TEST_F(RKFTest, PredictAndUpdate){
  ParaKF_->Step(meas_);
  auto RKF = dynamic_cast<ExtendedKF*> (ParaKF_);
  vector<float> Calc(mu_.data(),mu_.data()+mu_.rows()*mu_.cols());
  vector<float> Ans {1.02359,0.515336,5.76462,1.14404};

  vector<float>Calc2(Sigma_.data(),Sigma_.data()+Sigma_.rows()*Sigma_.cols());
  vector<float>Ans2{0.042377,0.0401995,0.0102694,-0.00952716,
                    0.0401995,0.042377,-0.00952716,0.0102694,
                    0.0102694,-0.00952716,45.6029,-45.513,
                    -0.00952716,0.0102694,-45.513,45.6029};

  for (int i = 0; i < Calc.size(); ++i){
    EXPECT_NEAR(Calc[i], Ans[i], 0.001) << "for i: " << i;
  }

  for (int i = 0; i < Calc2.size(); ++i){
    EXPECT_NEAR(Calc2[i], Ans2[i], 0.001) << "for i: " << i;
  }
}


class FusionTest : public :: testing::Test {
  protected:

  virtual void SetUp(){

  measL1.sensor_type_ = LASER;
  measL1.raw_measurements_ = VectorXd(2);
  measL1.raw_measurements_ << 0.312242687, 0.580339789;
  measL1.timestamp_ = 0;

  measR1.sensor_type_ = RADAR;
  measR1.raw_measurements_ = VectorXd(3);
  measR1.raw_measurements_ << 1.01489,0.554329,4.89281; 
  measR1.timestamp_ = measL1.timestamp_ + 50000;

  measL2.sensor_type_ = LASER;
  measL2.raw_measurements_ = VectorXd(2);
  measL2.raw_measurements_ << 1.17385,0.481073;
  measL2.timestamp_ = measR1.timestamp_ + 50000;

  measR2.sensor_type_ = RADAR;
  measR2.raw_measurements_ = VectorXd(3);
  measR2.raw_measurements_ << 1.04751,0.38924,4.51132;
  measR2.timestamp_ = measL2.timestamp_ + 50000;
  }

  virtual void TearDown(){};

  MeasurementPackage measL1;
  MeasurementPackage measL2;

  MeasurementPackage measR1;
  MeasurementPackage measR2;
};

TEST_F(FusionTest, FusionLaserLKF){
  FusionKF FKF(CONSTANT_VELOCITY);
  FKF.AddLaserLKF();

  FKF.ProcessMeasurement(measL1);

  auto Mu = FKF.GetMu();
  auto Sg = FKF.GetSigma();
  vector<float> Calc(Mu.data(),Mu.data()+Mu.rows()*Mu.cols());
  vector<float> Ans {0.312243,0.58034,0,0};    

  vector<float>Calc2(Sg.data(),Sg.data()+Sg.rows()*Sg.cols());
  vector<float> Ans2 {0.0220049,0,0,0,
                      0,0.0220049,0,0,
                      0,0,1000,0,
                      0,0,0,1000};

  for (int i = 0; i < Calc.size(); ++i){
    EXPECT_NEAR(Calc[i], Ans[i], 0.001) << "for i: " << i;
  }

  for (int i = 0; i < Calc2.size(); ++i){
    EXPECT_NEAR(Calc2[i], Ans2[i], 0.001) << "for i: " << i;
  }

  FKF.ProcessMeasurement(measL2);
  Mu = FKF.GetMu();
  Sg = FKF.GetSigma();
  vector<float>Calc3(Mu.data(),Mu.data()+Mu.rows()*Mu.cols());
  vector<float>Ans3{1.17192,0.481295,8.57809,-0.988292};    

  vector<float>Calc4(Sg.data(),Sg.data()+Sg.rows()*Sg.cols());
  vector<float>Ans4{0.0224496,0,0.224008,0,
                    0,0.0224496,0,0.224008,
                    0.224008,0,4.45347,0,
                    0,0.224008,0,4.45347};    

  for (int i = 0; i < Calc3.size(); ++i){
    EXPECT_NEAR(Calc3[i], Ans3[i], 0.001) << "for i: " << i;
  }

  for (int i = 0; i < Calc4.size(); ++i){
    EXPECT_NEAR(Calc4[i], Ans4[i], 0.001) << "for i: " << i;
  }
}

TEST_F(FusionTest, FusionRadarEKF){
  FusionKF FKF(CONSTANT_VELOCITY);
  FKF.AddRadarEKF();

  FKF.ProcessMeasurement(measR1);

  auto Mu = FKF.GetMu();
  auto Sg = FKF.GetSigma();
  vector<float> Calc(Mu.data(),Mu.data()+Mu.rows()*Mu.cols());
  vector<float> Ans {0.870819,0.547247,4.29365,2.34518};    

  vector<float>Calc2(Sg.data(),Sg.data()+Sg.rows()*Sg.cols());
  vector<float> Ans2 {0.0638724,0.0342303,0,0,
                      0.0342303,0.0198985,0,0,
                      0,0,229.849,-420.653,
                      0,0,-420.653,770.241};

  for (int i = 0; i < Calc.size(); ++i){
    EXPECT_NEAR(Calc[i], Ans[i], 0.001) << "for i: " << i;
  }

  for (int i = 0; i < Calc2.size(); ++i){
    EXPECT_NEAR(Calc2[i], Ans2[i], 0.001) << "for i: " << i;
  }

  FKF.ProcessMeasurement(measR2);
  Mu = FKF.GetMu();
  Sg = FKF.GetSigma();
  vector<float>Calc3(Mu.data(),Mu.data()+Mu.rows()*Mu.cols());
  vector<float>Ans3{1.21447,0.460064,5.2626,0.11902};    

  vector<float>Calc4(Sg.data(),Sg.data()+Sg.rows()*Sg.cols());
  vector<float>Ans4{0.0322415,0.018169,-0.000717736,0.00503674,
                    0.018169,0.0130109,-0.0123944,0.0239852,
                    -0.000717736,-0.0123944,0.134095,-0.128462,
                    0.00503674,0.0239852,-0.128462,0.282386};                    
  for (int i = 0; i < Calc3.size(); ++i){
    EXPECT_NEAR(Calc3[i], Ans3[i], 0.001) << "for i: " << i;
  }

  for (int i = 0; i < Calc4.size(); ++i){
    EXPECT_NEAR(Calc4[i], Ans4[i], 0.001) << "for i: " << i;
  }
}

TEST_F(FusionTest, FusionLaserAndRadar){
  FusionKF FKF(CONSTANT_VELOCITY);
  FKF.AddRadarEKF();
  FKF.AddLaserLKF();
  FKF.ProcessMeasurement(measL1);
  FKF.ProcessMeasurement(measR1);
  FKF.ProcessMeasurement(measL2);
  FKF.ProcessMeasurement(measR2);

  auto Mu = FKF.GetMu();
  auto Sg = FKF.GetSigma();
  vector<float>Calc(Mu.data(),Mu.data()+Mu.rows()*Mu.cols());
  vector<float>Ans{1.09685,0.586652,4.81974,2.05109};

  vector<float>Calc2(Sg.data(),Sg.data()+Sg.rows()*Sg.cols());
  vector<float>Ans2{0.0044511,0.00215911,0.0186872,-0.0177024,
                    0.00215911,0.00313799,0.00460727,0.000653721,
                    0.0186872,0.00460727,0.125729,-0.0910125,
                    -0.0177024,0.000653721,-0.0910125,0.194103};  

  for (int i = 0; i < Calc.size(); ++i){
    EXPECT_NEAR(Calc[i], Ans[i], 0.001) << "for i: " << i;
  }

  for (int i = 0; i < Calc2.size(); ++i){
    EXPECT_NEAR(Calc2[i], Ans2[i], 0.001) << "for i: " << i;
  }
}


TEST(UnscentedTest, SimpleRun){
  FusionKF FKF(CONSTANT_TURNRATE_VELOCITY);
  FKF.AddRadarUKF();
  MeasurementPackage meas_in;
  meas_in.timestamp_ = 0;
  meas_in.raw_measurements_ = VectorXd(3);
  meas_in.raw_measurements_ << 5.9214,0.2187,2.0062;
  
  FKF.ProcessMeasurement(meas_in);
  //cout << "Mu\n" << FKF.GetMu() << endl;

  MeasurementPackage meas_in2;
  meas_in.timestamp_ = 100000;
  meas_in.raw_measurements_ = VectorXd(3);
  meas_in.raw_measurements_ << 5.9214,0.2187,2.0062;
}

