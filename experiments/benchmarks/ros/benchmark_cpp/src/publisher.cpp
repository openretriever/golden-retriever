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

std::vector<size_t> kSizes {
    8,
    64,
    512,
    10 * 512,
    100 * 512,
    1000 * 512,
    10000 * 512,
    8
};
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
