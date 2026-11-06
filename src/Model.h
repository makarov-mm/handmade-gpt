// The transformer itself: parameters, activations, hand-written forward and
// backward passes, Adam, and sampling. No Qt in this file.
#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace gpt {

// Architecture. Everything except the vocabulary is fixed at compile time
// so the hot loops see constants.
struct Config {
    int vocab = 0;                 // from the corpus
    static constexpr int width = 64;      // model width d
    static constexpr int heads = 2;
    static constexpr int headDim = width / heads;
    static constexpr int context = 64;    // sequence length T
    static constexpr int hidden = 256;    // MLP hidden size
};

// All parameters live in one flat vector; the offsets name its regions,
// so the gradient buffer and the Adam moments share the layout.
struct Params {
    std::vector<float> w;
    int emb = 0, pos = 0, ln1g = 0, ln1b = 0, wq = 0, wk = 0, wv = 0, wo = 0,
        ln2g = 0, ln2b = 0, w1 = 0, b1 = 0, w2 = 0, b2 = 0, lnfg = 0, lnfb = 0,
        wu = 0, bu = 0, total = 0;

    void layout(const Config &cfg);
    void init(const Config &cfg, std::uint32_t seed);
};

// Activations of one sequence, kept for backprop, plus the scratch buffers
// the backward pass needs. One instance per worker thread; allocated once.
struct Activations {
    std::vector<float> x, h1, x1hat, q, k, v, A, ctxv, attn, a,
                       h2, ahat, m1, gelu, m2, b, bhat, hf, probs;
    std::vector<float> mu1, is1, mu2, is2, muf, isf; // layer-norm stats per position
    // backward scratch
    std::vector<float> dlogits, dhf, db, dgelu, dm1, da, dh2, dctx, dq, dk, dv, dh1, dArow;

    void allocate(const Config &cfg);
};

class Model {
public:
    Config cfg;
    Params par;
    std::vector<float> adamM, adamV;
    int step = 0;

    // Builds the vocabulary from the text, tokenizes it and initializes the
    // weights deterministically.
    void build(std::string_view text);

    [[nodiscard]] int vocab() const { return cfg.vocab; }
    [[nodiscard]] int paramCount() const { return par.total; }
    [[nodiscard]] std::span<const std::uint8_t> tokens() const { return m_tokens; }
    [[nodiscard]] char itoc(int i) const { return m_itoc[static_cast<std::size_t>(i)]; }

    // Forward one window of `context` tokens: next-token probabilities for
    // every position in ac.probs, everything backward needs in `ac`.
    void forward(const std::uint8_t *seq, Activations &ac) const;

    // Mean next-token cross-entropy of the last forward; `seq` must hold
    // context + 1 tokens, the extra one being the target of the final position.
    [[nodiscard]] float loss(const std::uint8_t *seq, const Activations &ac) const;

    // Backward one window (same `seq` as loss); accumulates into `grad`,
    // which has the layout of par.w.
    void backward(const std::uint8_t *seq, Activations &ac, float *grad) const;

    // One Adam update with bias correction.
    void adam(const float *grad, float lr);

    // Samples text from the model, primed with a random corpus window.
    [[nodiscard]] std::string generate(int nChars, float temperature, std::uint32_t seed) const;

private:
    std::vector<char> m_itoc;
    int m_ctoi[256] = {};
    std::vector<std::uint8_t> m_tokens;
};

} // namespace gpt
