/**
 * C++ IK Solver Node for Retriever
 *
 * Receives: pose (6 floats)
 * Outputs: joints (6 floats)
 */

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

// Dora C++ API
extern "C" {

enum DoraEventType {
  DoraStop,
  DoraInput,
  DoraInputClosed,
  DoraError,
  DoraUnknown,
};

void *init_dora_context_from_env();
void free_dora_context(void *context);
void *dora_next_event(void *context);
void free_dora_event(void *event);
DoraEventType read_dora_event_type(void *event);
void read_dora_input_id(void *event, uint8_t **out_ptr, size_t *out_len);
void read_dora_input_data(void *event, uint8_t **out_ptr, size_t *out_len);
int dora_send_output(void *context, const char *id_ptr, size_t id_len,
                     const uint8_t *data_ptr, size_t data_len);
}

std::vector<float> deserialize_f32_array(const uint8_t *data, size_t len) {
  std::vector<float> values;
  size_t count = len / 4;
  values.resize(count);
  if (count > 0) {
    std::memcpy(values.data(), data, len);
  }
  return values;
}

void serialize_f32_array(const std::vector<float> &values,
                         std::vector<uint8_t> &out) {
  size_t bytes = values.size() * 4;
  out.resize(bytes);
  std::memcpy(out.data(), values.data(), bytes);
}

int main() {
  std::cout << "[cpp-ik-solver] Starting...\n";

  void *context = init_dora_context_from_env();
  if (!context) {
    std::cerr << "[cpp-ik-solver] Failed to init dora context\n";
    return 1;
  }

  while (true) {
    void *event = dora_next_event(context);
    if (!event)
      break;

    DoraEventType type = read_dora_event_type(event);

    if (type == DoraInput) {
      uint8_t *id_ptr;
      size_t id_len;
      read_dora_input_id(event, &id_ptr, &id_len);
      std::string id(reinterpret_cast<const char *>(id_ptr), id_len);

      if (id == "pose") {
        uint8_t *data_ptr;
        size_t data_len;
        read_dora_input_data(event, &data_ptr, &data_len);

        auto pose = deserialize_f32_array(data_ptr, data_len);

        if (pose.size() >= 6) {
          float x = pose[0];
          float y = pose[1];
          float z = pose[2];

          // Mock IK
          float j1 = std::atan2(y, x);
          float j2 = z * 2.0f;
          float j3 = x + y;

          std::vector<float> joints = {j1, j2, j3, 0.0f, 0.0f, 0.0f};

          std::vector<uint8_t> output;
          serialize_f32_array(joints, output);

          dora_send_output(context, "joints", 6, output.data(), output.size());
        }
      }
    } else if (type == DoraStop) {
      free_dora_event(event);
      break;
    }
    free_dora_event(event);
  }

  free_dora_context(context);
  return 0;
}
