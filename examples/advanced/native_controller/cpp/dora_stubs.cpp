/**
 * Dora C++ API Stubs
 *
 * Provides symbol definitions for Dora API functions to satisfy the linker
 * when building without the actual dora shared library.
 */

#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

extern "C" {
// Stub handles
struct DoraNode {
  int id;
};
struct DoraEvent {
  char type[16];
  char id[32];
  std::vector<uint8_t> data;
};

// Global stub state to simulate some events
static int stub_counter = 0;

void *init_dora_node() {
  std::cout << "[WARN] Running with DORA STUBS (Real Dora lib not linked)\n";
  return new DoraNode{1};
}

void *dora_next_event(void *node) {
  if (!node)
    return nullptr;

  // Simulate a few events then stop
  stub_counter++;
  auto *event = new DoraEvent();

  if (stub_counter < 50) {
    std::strcpy(event->type, "INPUT");
    std::strcpy(event->id, "pose");
    // Mock pose data: [0.5, 0.0, 0.4, 0, 3.14, 0] * 4 bytes
    float mock_pose[] = {0.5f, 0.0f, 0.4f, 0.0f, 3.14f, 0.0f};
    size_t bytes = sizeof(mock_pose);
    event->data.resize(bytes);
    std::memcpy(event->data.data(), mock_pose, bytes);
  } else {
    std::strcpy(event->type, "STOP");
  }

  // Add artificial delay to simulate rate
  // struct timespec req = {0, 1000 * 1000 * 20}; // 20ms
  // nanosleep(&req, NULL);

  return event;
}

const char *event_type(void *event) {
  return static_cast<DoraEvent *>(event)->type;
}

const char *event_id(void *event) {
  return static_cast<DoraEvent *>(event)->id;
}

const uint8_t *event_data(void *event) {
  return static_cast<DoraEvent *>(event)->data.data();
}

size_t event_data_len(void *event) {
  return static_cast<DoraEvent *>(event)->data.size();
}

int send_dora_output(void *node, const char *id, const uint8_t *data,
                     size_t len) {
  // Just print output in stub mode
  // std::cout << "[STUB] Sending output '" << id << "' (" << len << "
  // bytes)\n";
  return 0;
}

void free_dora_event(void *event) { delete static_cast<DoraEvent *>(event); }
}
