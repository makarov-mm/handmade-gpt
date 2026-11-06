// Mini-batch training loop: random context windows, data-parallel forward
// and backward over a persistent worker pool with per-thread gradient
// buffers, reduction, Adam step. No Qt in this file.
#pragma once

#include "Model.h"

#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <random>
#include <string_view>
#include <thread>
#include <vector>

namespace gpt {

class Trainer {
public:
    Model model;
    float lr = 3e-3f;
    int batch = 12;
    float lastLoss = 0.f;
    std::vector<float> lossHistory;
    std::vector<float> lastAttn;   // attention maps of the first sequence of the last step, for the canvas
    float tokensPerSec = 0.f;

    Trainer();
    ~Trainer();
    Trainer(const Trainer &) = delete;
    Trainer &operator=(const Trainer &) = delete;

    // Rebuilds the model on new text and clears the training state.
    void build(std::string_view text);

    // Runs `nSteps` optimizer steps on the calling thread (plus the pool).
    void trainSteps(int nSteps);

    [[nodiscard]] int threadCount() const { return static_cast<int>(m_grads.size()); }

private:
    // One worker per hardware thread (at least two); each owns a gradient
    // buffer and an activation set, so a step allocates nothing.
    std::vector<std::vector<float>> m_grads;
    std::vector<Activations> m_acts;
    std::vector<std::thread> m_workers;
    std::vector<int> m_offsets;      // window start of every batch element
    std::vector<float> m_losses;     // loss of every batch element
    std::mt19937 m_rng{0xF17E55};

    // pool synchronization
    std::mutex m_mutex;
    std::condition_variable m_wake, m_done;
    std::uint64_t m_generation = 0;  // bumped once per step
    int m_pending = 0;               // workers still busy in this generation
    bool m_quit = false;

    void workerLoop(int index);
    void runSlice(int index);        // forward + backward of batch elements index, index + threads, ...
    void runStep();                  // one step over all workers, including this thread's slice
};

} // namespace gpt
