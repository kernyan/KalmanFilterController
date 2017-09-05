#ifndef PARAMETRIC_KF_H_
#define PARAMETRIC_KF_H_  
#include "Eigen/Dense"
#include "measurement_package.h"
#include <functional>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;

class ParametricKF {

public:

	virtual ~ParametricKF(){};
	virtual void Step(MeasurementPackage &meas_in) = 0;

protected:

	virtual void CalculateMuBar() = 0;
	virtual void CalculateSigmaBar() = 0;

	virtual MatrixXd CalculateMeasurementVar() = 0; // S
	virtual VectorXd CalculatePredictedMeasurement() = 0; // Zhat
};


class LinearKF : public ParametricKF {

public:

	LinearKF(VectorXd &Mu_in, MatrixXd &Sigma_in, long long &t_in);
	void Initialize(MatrixXd &TransFunc_in,
			MatrixXd &TransSigma_in,
			MatrixXd &MeasFunc_in,
			MatrixXd &MeasSigma_in);
  virtual void Step(MeasurementPackage &meas_in) override;
  VectorXd GetMuBar() const {return MuBar_;};
  MatrixXd GetSigmaBar() const {return SigmaBar_;};

protected:

	virtual void CalculateMuBar() override final;
	virtual void CalculateSigmaBar() override final;
	virtual MatrixXd CalculateMeasurementVar() override final;
	virtual VectorXd CalculatePredictedMeasurement() override final;
  void FirstTimeStep(MeasurementPackage &meas_in);

	// member variables

	VectorXd &Mu_;    // shared with all filters 
	MatrixXd &Sigma_; // shared with all filters 
  long long &previous_time_; // shared with all filters 

	VectorXd MuBar_;
	MatrixXd SigmaBar_;
	MatrixXd At_; // TransitionFunction
	MatrixXd Rt_; // TransitionSigma
	MatrixXd Ct_; // MeasurementFunction
	MatrixXd Qt_; // MeasurementSigma
};

class LaserLKF : public LinearKF {

  public:
    
    LaserLKF(VectorXd &Mu_in, MatrixXd &Sigma_in, long long &t_in);
    virtual void Step(MeasurementPackage &meas_in) override final;
    void Initialize();

  private:
  
    const SensorType Sensor_ = LASER;
    float noise_ax_;
    float noise_ay_;
};


using Func = function<VectorXd (VectorXd)>;

class ExtendedKF : public ParametricKF{
  
public:

  ExtendedKF(VectorXd &Mu_in, MatrixXd &Sigma_in, long long &t_in);

	void Initialize(Func &g_in,
			MatrixXd &G_in,
      MatrixXd &Rt_in,
			Func &h_in,
      function<MatrixXd (VectorXd)> &H_in,
			MatrixXd &Qt_in);
  virtual void Step(MeasurementPackage &meas_in) override;
  VectorXd GetMuBar() const {return MuBar_;};
  MatrixXd GetSigmaBar() const {return SigmaBar_;};

    
protected:

	virtual void CalculateMuBar() override final;
	virtual void CalculateSigmaBar() override final;
	virtual MatrixXd CalculateMeasurementVar() override final;
	virtual VectorXd CalculatePredictedMeasurement() override final;
  MatrixXd CalculatePredJacobian() const;
  MatrixXd CalculateMeasJacobian() const;

	// member variables

	VectorXd &Mu_;    // shared with all filters 
	MatrixXd &Sigma_; // shared with all filters 
  long long &previous_time_; // shared with all filters 
  
	VectorXd MuBar_;
	MatrixXd SigmaBar_;
	Func g_;      // TransitionFunction
  MatrixXd G_;  // Jacobian of TransitionFunction
	MatrixXd Rt_; // TransitionSigma
	Func h_;      // MeasurementFunction
  function<MatrixXd (VectorXd)> H_;  // Jacobian of MeasurementFunction
	MatrixXd Qt_; // MeasurementSigma
};


class RadarEKF : public ExtendedKF {

  public:
    
    RadarEKF(VectorXd &Mu_in, MatrixXd &Sigma_in, long long &t_in);
    virtual void Step(MeasurementPackage &meas_in) override final;
    void Initialize();

  private:
  
    const SensorType Sensor_ = RADAR;
    float noise_ax_;
    float noise_ay_;
};
#endif /* PARAMETRIC_KF_H_ */

