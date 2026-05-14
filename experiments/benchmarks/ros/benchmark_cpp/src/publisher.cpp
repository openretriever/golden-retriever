#include <chrono>
#include <memory>
#include <random>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp_components/register_node_macro.hpp"
#include "std_msgs/msg/u_int64_multi_array.hpp"

/**
 *  Publisher node for ROS 2 C++ benchmarking.
 */

// NOTE: kSizes defines the number of uint64 elements in the payload.
// Since each uint64 is 8 bytes, the actual payload size in bytes is: Element Count * 8.
// Matches SIZES in benchmark_python/publisher.py: [2^i for i in range(6, 25)].
//   i=6  -> 2^6  = 64 elements   -> 512 Bytes
//   i=20 -> 2^20 = 1M elements   -> 8 MB
//   i=24 -> 2^24 = 16M elements  -> 128 MB
std::vector<size_t> kSizes = []() {
    std::vector<size_t> sizes;
    sizes.reserve(25 - 6);
    for (size_t i = 6; i < 25; ++i) {
        sizes.push_back(static_cast<size_t>(1) << i);
    }
    return sizes;
}();
static constexpr size_t kNumPointsPerSize {100};
static constexpr std::chrono::milliseconds kDataRate {50};


class MinimalPublisher : public rclcpp::Node
{
public:
  explicit MinimalPublisher(const rclcpp::NodeOptions & options)
  : Node("minimal_publisher", options), i_(0), j_(0), rng_(std::random_device{}()), dist_(0, 254)
  {
    publisher_ = this->create_publisher<std_msgs::msg::UInt64MultiArray>("topic", 10);
    auto timer_callback =
      [this]() -> void {
        auto message = std::make_unique<std_msgs::msg::UInt64MultiArray>();
        message->data.resize(kSizes.at(i_));
        for (auto& elem : message->data)
        {
            elem = dist_(rng_);
        }
        message->data[0] = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()
        ).count();
        // RCLCPP_INFO(this->get_logger(), "Publishing message of size %lu", kSizes.at(i_));
        this->publisher_->publish(std::move(message));

        if (j_ == kNumPointsPerSize)
        {
            ++i_;
            j_ = 0;
            if (i_ >= kSizes.size())
            {
                RCLCPP_INFO(this->get_logger(), "Benchmarking data collection complete!");
                timer_->cancel();
            }
        } else {
            ++j_;
        }
      };
    timer_ = this->create_wall_timer(kDataRate, timer_callback);
  }

private:
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Publisher<std_msgs::msg::UInt64MultiArray>::SharedPtr publisher_;
  size_t i_;
  size_t j_;
  std::mt19937_64 rng_;
  std::uniform_int_distribution<uint64_t> dist_;
};

RCLCPP_COMPONENTS_REGISTER_NODE(MinimalPublisher)

#ifndef BENCHMARK_CPP_AS_COMPONENT
int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<MinimalPublisher>(rclcpp::NodeOptions()));
  rclcpp::shutdown();
  return 0;
}
#endif
