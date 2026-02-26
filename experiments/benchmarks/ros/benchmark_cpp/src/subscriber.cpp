#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/u_int64_multi_array.hpp"

static std::string kName{"ROS 2 C++"};
static std::string kPlatform{"COMPUTER_PERF"};
static std::string kOutputFilePath{"experiments/benchmarks/results/ros_cpp_benchmark_results.csv"};
static constexpr size_t kNumPointsPerSize {100};


class MinimalSubscriber : public rclcpp::Node
{
public:
  MinimalSubscriber()
  : Node("minimal_subscriber"), n_(0), file_(kOutputFilePath)
  {
    writeHeader();
    latencies_.reserve(kNumPointsPerSize);

    auto topic_callback =
      [this](std_msgs::msg::UInt64MultiArray::UniquePtr msg) -> void {
        // RCLCPP_INFO(this->get_logger(), "Received message of size: %lu", msg->data.size());
        const auto t_received = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()
        ).count();
        const auto t_send = msg->data.at(0);
        const auto length = msg->data.size() * 8;
        if (length != cur_size_) {
            if (n_ > 0) {
                writeRow(cur_size_, latencies_);
            }
            cur_size_ = length;
            n_ = 0;
            latencies_.clear();
            latencies_.reserve(kNumPointsPerSize);
        }
        latencies_.push_back(t_received - t_send);
        ++n_;
    
    };
    subscription_ =
      this->create_subscription<std_msgs::msg::UInt64MultiArray>("topic", 10, topic_callback);
  }

  void writeHeader() {
    file_ << "name,platform,size,latency_ns\n";
  }

  void writeRow(size_t cur_size, const std::vector<size_t>& latencies) {
    file_ << kName << "," << kPlatform << "," << cur_size << ",\"[";
    for (size_t i = 0; i < latencies.size(); ++i) {
        file_ << std::to_string(latencies.at(i));
        if (i < latencies.size() - 1) {
            file_ << ",";
        }
    }
    file_ << "]\"\n";
  }

  ~MinimalSubscriber() {
    if (file_.is_open()) {
        file_.close();
    }
  }

private:
  rclcpp::Subscription<std_msgs::msg::UInt64MultiArray>::SharedPtr subscription_;
  size_t n_;
  size_t cur_size_;
  std::vector<size_t> latencies_;  // nanoseconds
  std::ofstream file_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<MinimalSubscriber>());
  rclcpp::shutdown();
  return 0;
}