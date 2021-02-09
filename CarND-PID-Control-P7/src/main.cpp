#include "PID.h"
#include "json.hpp"
#include <iostream>
#include <math.h>
#include <numeric>
#include <string>
#include <uWS/uWS.h>
#include <vector>

// for convenience
using nlohmann::json;
using std::string;
using std::vector;

// For converting back and forth between radians and degrees.
constexpr int kMinIterations = 50;
constexpr int kMaxIterations = 600;
constexpr double kTwiddleTolerance = 1e-5;
constexpr double pi() { return M_PI; }
double deg2rad(double x) { return x * pi() / 180; }
double rad2deg(double x) { return x * 180 / pi(); }
double SumVector(const vector<double> vec) { return std::accumulate(vec.begin(), vec.end(), 0.0); }

// Checks if the SocketIO event has JSON data.
// If there is data the JSON object in string format will be returned,
// else the empty string "" will be returned.
string hasData(string s)
{
  auto found_null = s.find("null");
  auto b1 = s.find_first_of("[");
  auto b2 = s.find_last_of("]");
  if (found_null != string::npos)
  {
    return "";
  }
  else if (b1 != string::npos && b2 != string::npos)
  {
    return s.substr(b1, b2 - b1 + 1);
  }
  return "";
}

struct Twiddle
{
  bool stop = false;
  bool is_first_loop = true;
  bool is_active = false;
  int i = 0;
  int n = 0;
  double error = 0.0;
  double mean_error = 0.0;
  double best_error = 1e5;
  vector<double> dp{1, 0.01, 1};
  vector<bool> increase{true, true, true};

  void PrintDP() { printf("DP=[%.5f|%.5f|%.5f] n=%i i=%i  +++ ", dp[0], dp[1], dp[2], n, i); }
};

bool CheckTrainingStatus(Twiddle &twiddle, const double &speed, const double &angle)
{

  bool restart_twiddle = false;
  auto &n = twiddle.n;
  bool is_any_thresholds_exceeded = (speed < 1.0 || deg2rad(angle) > 1.0 || deg2rad(angle) < -1.0);
  if ((n > kMinIterations && is_any_thresholds_exceeded) || n > kMaxIterations)
  {
    restart_twiddle = true;
    std::cout << "Twiddle has been activated/restarted" << std::endl;
    std::cout << "Speed: " << speed << std::endl;
    std::cout << "Angle (degree): " << (angle) << std::endl;
    std::cout << "Angle (rad): " << deg2rad(angle) << std::endl;
  }
  return restart_twiddle;
}

void Train(Twiddle &twiddle, PID &pid, const double &mean_error)
{
  auto &i = twiddle.i;
  auto &dp = twiddle.dp;
  auto &best_error = twiddle.best_error;
  auto &done = twiddle.increase;
  if (twiddle.is_first_loop)
  {
    twiddle.is_first_loop = false;
  }
  else
  {
    if (mean_error < best_error)
    {
      // New best error found then increase and go to next dp_i
      best_error = mean_error;
      dp[i] *= 1.1;
      done[i] = true;
    }
    else if (done[i])
    {
      pid.AddParameterByIndex(-2 * dp[i], i);
      done[i] = false;
    }
    else
    {
      pid.AddParameterByIndex(dp[i], i);
      dp[i] *= 0.9;
      done[i] = true;
    }

  } // check if first loop

  if (done[i])
  {
    // Reset to default: false
    // done[i] = false;

    // Go to the next dp
    i = (i == 2) ? 0 : i + 1;

    // Add dp prior the next loop
    pid.AddParameterByIndex(dp[i], i);
  }
}

int main()
{
  uWS::Hub h;

  PID pid;
  Twiddle twiddle;

  /**
   * Initialize the pid variable.
   */
  double kp = 0.53;
  double ki = 0.005;
  double kd = 3.7;
  pid.Init(kp, ki, kd);

  // Train with twiddle (=false) or run without twiddle (=true)
  twiddle.stop = true;

  h.onMessage([&pid, &twiddle](uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length,
                               uWS::OpCode opCode) {
    // "42" at the start of the message means there's a websocket message event.
    // The 4 signifies a websocket message
    // The 2 signifies a websocket event
    if (length && length > 2 && data[0] == '4' && data[1] == '2')
    {
      auto s = hasData(string(data).substr(0, length));

      if (s != "")
      {
        auto j = json::parse(s);

        string event = j[0].get<string>();

        if (event == "telemetry")
        {
          // j[1] is the data JSON object
          double cte = std::stod(j[1]["cte"].get<string>());
          double speed = std::stod(j[1]["speed"].get<string>());
          double angle = std::stod(j[1]["steering_angle"].get<string>());
          double steer_value;
          /**
           * TODO: Calculate steering value here, remember the steering value is
           *   [-1, 1].
           * NOTE: Feel free to play around with the throttle and speed.
           *   Maybe use another PID controller to control the speed!
           */

          // Check if twiddle should be restarted
          if (twiddle.stop)
          {
            std::cout << " => Twiddle has been stopped!" << std::endl;
          }
          else
          {
            twiddle.is_active = CheckTrainingStatus(twiddle, speed, angle);

            // Creating aliases
            auto &n = twiddle.n;
            auto &error = twiddle.error;
            auto &mean_error = twiddle.mean_error;
            auto &dp = twiddle.dp;

            // Init values
            n++;
            error += cte * cte;
            mean_error = error / n;

            // Twiddle
            if (twiddle.is_active && SumVector(dp) > kTwiddleTolerance)
            {
              Train(twiddle, pid, mean_error);
              n = 0;
              error = 0;
              mean_error = 0;
              twiddle.is_active = false;

              // Reset simulator
              std::string msg = "42[\"reset\",{}]";
              ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
            }
            else if (SumVector(dp) < kTwiddleTolerance)
            {
              std::cout << " => Twiddle accuracy reached: " << SumVector(dp) << std::endl;
              twiddle.stop = true;
            } // twiddle active and accuracy can be increased
          }

          // Run car
          pid.UpdateError(cte);
          steer_value = pid.TotalError();

          // DEBUG
          twiddle.PrintDP();
          printf(" {%.1f}", angle);
          pid.PrintK();

          json msgJson;
          msgJson["steering_angle"] = steer_value;
          msgJson["throttle"] = 0.3;
          auto msg = "42[\"steer\"," + msgJson.dump() + "]";
          // std::cout << msg << std::endl;
          ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
        } // end "telemetry" if
      }
      else
      {
        // Manual driving
        string msg = "42[\"manual\",{}]";
        ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
      }
    } // end websocket message if
  }); // end h.onMessage

  h.onConnection([&h](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
    std::cout << "Connected!!!" << std::endl;
  });

  h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code,
                         char *message, size_t length) {
    ws.close();
    std::cout << "Disconnected" << std::endl;
  });

  int port = 4567;
  if (h.listen(port))
  {
    std::cout << "Listening to port " << port << std::endl;
  }
  else
  {
    std::cerr << "Failed to listen to port" << std::endl;
    return -1;
  }

  h.run();
}
