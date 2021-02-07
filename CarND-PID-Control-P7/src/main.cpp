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
constexpr double pi() { return M_PI; }
double deg2rad(double x) { return x * pi() / 180; }
double rad2deg(double x) { return x * 180 / pi(); }

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
  bool is_active_ = true;
  int i_ = 0;
  int n_ = 0;
  double error_ = 1e5;
  double best_error_ = 1e5;
  double mean_error_ = 0.0;
  vector<double> p_{0, 0, 0};
  vector<double> dp_{1, 1, 1};
  vector<bool> increase_{true, true, true};

  double SumDp() { return std::accumulate(dp_.begin(), dp_.end(), 0); }
  void PrintP() { std::cout << "P=" << p_[0] << "|" << p_[1] << "|" << p_[2] << std::endl; }
};

int main()
{
  uWS::Hub h;

  PID pid;
  Twiddle twiddle;
  /**
   * TODO: Initialize the pid variable.
   */
  pid.Init(0.1, 0.3, 0.0001);

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
          if (twiddle.n_ > 100 && speed <= 1.0)
          {
            twiddle.is_active_ = false;
            std::cout << "Warning: Twiddle has been deactivated" << std::endl;
          }

          // Twiddle
          if (twiddle.is_active_ && twiddle.SumDp() > 1e-9)
          {
            auto &i = twiddle.i_;
            auto &n = twiddle.n_;
            auto &p = twiddle.p_;
            auto &dp = twiddle.dp_;
            auto &error = twiddle.error_;
            auto &best_error = twiddle.best_error_;
            auto &mean_error = twiddle.mean_error_;
            auto &increase = twiddle.increase_;

            // Init values
            n++;
            error += cte * cte;
            mean_error = error / n;

            if (mean_error < best_error)
            {
              best_error = mean_error;
              dp[i] *= 1.1;
              increase[i] = true;
              i = (i == 3) ? 0 : i + 1;
            }
            else
            {
              if (increase[i])
              {
                p[i] -= 2 * dp[i];
                increase[i] = false;
              }
              else
              {
                pid.SetParameterByIndex(dp[i], i);
                dp[i] *= 0.9;
                increase[i] = true;
              }
            }

            if (increase[i])
            {
              pid.SetParameterByIndex(dp[i], i);
            }
            n = 0;
            error = 0;
            mean_error = 0;

          } // twiddle active and accuracy can be increased

          // Run car
          pid.UpdateError(cte);
          steer_value = pid.TotalError();

          // DEBUG
          twiddle.PrintP();
          pid.PrintK();
          std::cout << "CTE: " << cte << " Steering Value: " << steer_value
                    << std::endl;

          json msgJson;
          msgJson["steering_angle"] = steer_value;
          msgJson["throttle"] = 0.3;
          auto msg = "42[\"steer\"," + msgJson.dump() + "]";
          std::cout << msg << std::endl;
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
