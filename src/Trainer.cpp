#include "Trainer.h"

#include <algorithm>
#include <chrono>

namespace gpt {

Trainer::Trainer()
{
    const int threads = static_cast<int>(std::max(2u, std::thread::hardware_concurrency()));
    m_grads.resize(static_cast<std::size_t>(threads));
    m_acts.resize(static_cast<std::size_t>(threads));
    // The calling thread takes slice 0; the pool covers the rest.
    for (int i = 1; i < threads; ++i)
        m_workers.emplace_back([this, i] { workerLoop(i); });
}

Trainer::~Trainer()
{
    {
        std::lock_guard lock(m_mutex);
        m_quit = true;
    }
    m_wake.notify_all();
    for (std::thread &worker : m_workers)
        worker.join();
}

void Trainer::build(std::string_view text)
{
    model.build(text);
    for (std::vector<float> &g : m_grads)
        g.assign(static_cast<std::size_t>(model.paramCount()), 0.f);
    for (Activations &ac : m_acts)
        ac.allocate(model.cfg);
    lossHistory.clear();
    lastAttn.clear();
    lastLoss = 0.f;
    tokensPerSec = 0.f;
    m_rng.seed(0xF17E55);
}

void Trainer::workerLoop(int index)
{
    std::uint64_t seen = 0;
    for (;;) {
        {
            std::unique_lock lock(m_mutex);
            m_wake.wait(lock, [&] { return m_quit || m_generation != seen; });
            if (m_quit)
                return;
            seen = m_generation;
        }
        runSlice(index);
        {
            std::lock_guard lock(m_mutex);
            if (--m_pending == 0)
                m_done.notify_one();
        }
    }
}

void Trainer::runSlice(int index)
{
    const int threads = threadCount();
    std::vector<float> &grad = m_grads[static_cast<std::size_t>(index)];
    Activations &ac = m_acts[static_cast<std::size_t>(index)];
    std::ranges::fill(grad, 0.f);
    const std::uint8_t *tokens = model.tokens().data();
    for (int b = index; b < batch; b += threads) {
        const std::uint8_t *seq = tokens + m_offsets[static_cast<std::size_t>(b)];
        model.forward(seq, ac);
        m_losses[static_cast<std::size_t>(b)] = model.loss(seq, ac);
        model.backward(seq, ac, grad.data());
        if (b == 0)
            lastAttn.assign(ac.A.begin(), ac.A.end());
    }
}

void Trainer::runStep()
{
    {
        std::lock_guard lock(m_mutex);
        m_pending = threadCount() - 1;
        ++m_generation;
    }
    m_wake.notify_all();
    runSlice(0);
    std::unique_lock lock(m_mutex);
    m_done.wait(lock, [&] { return m_pending == 0; });
}

void Trainer::trainSteps(int nSteps)
{
    const auto tokens = model.tokens();
    if (tokens.size() < static_cast<std::size_t>(Config::context) + 2)
        return; // nothing to sample windows from
    const int P = model.paramCount();
    const int threads = threadCount();
    std::uniform_int_distribution<int> starts(0, static_cast<int>(tokens.size()) - Config::context - 2);
    const auto clockStart = std::chrono::steady_clock::now();
    int tokensDone = 0;

    for (int s = 0; s < nSteps; ++s) {
        m_offsets.resize(static_cast<std::size_t>(batch));
        m_losses.assign(static_cast<std::size_t>(batch), 0.f);
        for (int &o : m_offsets) o = starts(m_rng);

        runStep();

        // reduce the per-thread gradients into slice 0, mean over the batch
        float *g0 = m_grads[0].data();
        for (int t = 1; t < threads; ++t) {
            const float *gt = m_grads[static_cast<std::size_t>(t)].data();
            for (int i = 0; i < P; ++i) g0[i] += gt[i];
        }
        const float invB = 1.f / static_cast<float>(batch);
        for (int i = 0; i < P; ++i) g0[i] *= invB;
        model.adam(g0, lr);

        float sum = 0.f;
        for (float l : m_losses) sum += l;
        lastLoss = sum / static_cast<float>(batch);
        lossHistory.push_back(lastLoss);
        if (lossHistory.size() > 4000)
            lossHistory.erase(lossHistory.begin(), lossHistory.begin() + 2000);
        tokensDone += batch * Config::context;
    }
    const float seconds = std::chrono::duration<float>(std::chrono::steady_clock::now() - clockStart).count();
    tokensPerSec = static_cast<float>(tokensDone) / std::max(1e-3f, seconds);
}

} // namespace gpt
