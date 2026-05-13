#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp_components/register_node_macro.hpp"
#include "std_msgs/msg/u_int64_multi_array.hpp"

static const std::string kPlatform{"COMPUTER_PERF"};
static const std::string kDefaultName{"ROS 2 C++"};
static const std::string kDefaultOutputFile{
    "experiments/benchmarks/results/ros_cpp_benchmark_results.csv"};
static constexpr size_t kNumPointsPerSize {100};


class MinimalSubscriber : public rclcpp::Node
{
public:
  explicit MinimalSubscriber(const rclcpp::NodeOptions & options)
  : Node("minimal_subscriber", options),
    n_(0),
    name_(this->declare_parameter<std::string>("benchmark_name", kDefaultName)),
    output_file_(this->declare_parameter<std::string>("output_file", kDefaultOutputFile))
  {
    file_.open(output_file_);
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
    file_.flush();
  }

  void writeRow(size_t cur_size, const std::vector<size_t>& latencies) {
    file_ << name_ << "," << kPlatform << "," << cur_size << ",\"[";
    for (size_t i = 0; i < latencies.size(); ++i) {
        file_ << std::to_string(latencies.at(i));
        if (i < latencies.size() - 1) {
            file_ << ",";
        }
    }
    file_ << "]\"\n";
    file_.flush();
  }

  ~MinimalSubscriber() {
    if (file_.is_open()) {
        if (n_ > 0) {
            writeRow(cur_size_, latencies_);
        }
        file_.close();
    }
  }

private:
  rclcpp::Subscription<std_msgs::msg::UInt64MultiArray>::SharedPtr subscription_;
  size_t n_;
  size_t cur_size_;
  std::vector<size_t> latencies_;  // nanoseconds
  std::string name_;
  std::string output_file_;
  std::ofstream file_;
};

RCLCPP_COMPONENTS_REGISTER_NODE(MinimalSubscriber)

#ifndef BENCHMARK_CPP_AS_COMPONENT
int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<MinimalSubscriber>(rclcpp::NodeOptions()));
  rclcpp::shutdown();
  return 0;
}
#endif