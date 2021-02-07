#ifndef PID_H
#define PID_H

class PID
{
public:
  /**
   * Constructor
   */
  PID();

  /**
   * Destructor.
   */
  virtual ~PID();

  /**
   * Initialize PID.
   * @param (Kp_, Ki_, Kd_) The initial PID coefficients
   */
  void Init(double Kp_, double Ki_, double Kd_);

  /**
   * Update the PID error variables given cross track error.
   * @param cte The current cross track error
   */
  void UpdateError(double cte);

  /**
   * Calculate the total PID error.
   * @output The total PID error
   */
  double TotalError();

  /**
   * Setter for any of the k-parameters according to idx
   */
  void SetParameterByIndex(double value, int index);

  /**
   * Print function for the k-parameters for debugging
   */
  void PrintK();

private:
  /**
   * PID Errors
   */
  double p_error_;
  double i_error_;
  double d_error_;

  /**
   * PID Coefficients
   */
  double kp_;
  double ki_;
  double kd_;
};

#endif // PID_H
