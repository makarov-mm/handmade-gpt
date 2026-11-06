#include "Model.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numbers>
#include <random>

namespace gpt {
namespace {

constexpr int T = Config::context;
constexpr int D = Config::width;
constexpr int H = Config::heads;
constexpr int DH = Config::headDim;
constexpr int HID = Config::hidden;
constexpr float kLnEps = 1e-5f;
constexpr float kGeluRoot = 0.7978845608f; // sqrt(2 / pi)
constexpr float kGeluCubic = 0.044715f;

// GELU, tanh approximation, and its derivative
float geluF(float x)
{
    const float u = kGeluRoot * (x + kGeluCubic * x * x * x);
    return 0.5f * x * (1.f + std::tanh(u));
}

float geluD(float x)
{
    const float u = kGeluRoot * (x + kGeluCubic * x * x * x);
    const float t = std::tanh(u);
    const float du = kGeluRoot * (1.f + 3.f * kGeluCubic * x * x);
    return 0.5f * (1.f + t) + 0.5f * x * (1.f - t * t) * du;
}

// y = x @ W            x: [rows, in]   W: [in, out]   y: [rows, out]
void matmul(const float *x, const float *W, float *y, int rows, int in, int out)
{
    for (int r = 0; r < rows; ++r) {
        const float *xr = x + r * in;
        float *yr = y + r * out;
        std::memset(yr, 0, sizeof(float) * static_cast<std::size_t>(out));
        for (int i = 0; i < in; ++i) {
            const float xi = xr[i];
            const float *wr = W + i * out;
            for (int o = 0; o < out; ++o) yr[o] += xi * wr[o];
        }
    }
}

// dx += dy @ W^T ;  dW += x^T @ dy
void matmulBack(const float *x, const float *W, const float *dy,
                float *dx, float *dW, int rows, int in, int out)
{
    for (int r = 0; r < rows; ++r) {
        const float *xr = x + r * in;
        const float *dyr = dy + r * out;
        float *dxr = dx + r * in;
        for (int i = 0; i < in; ++i) {
            const float *wr = W + i * out;
            float *dwr = dW + i * out;
            const float xi = xr[i];
            float acc = 0.f;
            for (int o = 0; o < out; ++o) {
                acc += dyr[o] * wr[o];
                dwr[o] += xi * dyr[o];
            }
            dxr[i] += acc;
        }
    }
}

// y = LN(x) * g + b per row; keeps xhat and the inverse std for backward
void layerNorm(const float *x, const float *g, const float *b,
               float *y, float *xhat, float *muOut, float *isOut)
{
    for (int t = 0; t < T; ++t) {
        const float *xr = x + t * D;
        float mu = 0.f;
        for (int i = 0; i < D; ++i) mu += xr[i];
        mu /= D;
        float var = 0.f;
        for (int i = 0; i < D; ++i) { const float c = xr[i] - mu; var += c * c; }
        const float is = 1.f / std::sqrt(var / D + kLnEps);
        muOut[t] = mu;
        isOut[t] = is;
        for (int i = 0; i < D; ++i) {
            const float xh = (xr[i] - mu) * is;
            xhat[t * D + i] = xh;
            y[t * D + i] = xh * g[i] + b[i];
        }
    }
}

void layerNormBack(const float *xhat, const float *g, const float *is,
                   const float *dy, float *dx, float *dg, float *db)
{
    for (int t = 0; t < T; ++t) {
        const float *xh = xhat + t * D;
        const float *dyr = dy + t * D;
        float s1 = 0.f, s2 = 0.f;
        for (int i = 0; i < D; ++i) {
            const float dyg = dyr[i] * g[i];
            s1 += dyg;
            s2 += dyg * xh[i];
            dg[i] += dyr[i] * xh[i];
            db[i] += dyr[i];
        }
        s1 /= D;
        s2 /= D;
        float *dxr = dx + t * D;
        for (int i = 0; i < D; ++i)
            dxr[i] += (dyr[i] * g[i] - s1 - xh[i] * s2) * is[t];
    }
}

// In-place softmax over the first `n` entries of a row
void softmaxRow(float *row, int n)
{
    float mx = -1e30f;
    for (int i = 0; i < n; ++i) mx = std::max(mx, row[i]);
    float sum = 0.f;
    for (int i = 0; i < n; ++i) { row[i] = std::exp(row[i] - mx); sum += row[i]; }
    const float inv = 1.f / sum;
    for (int i = 0; i < n; ++i) row[i] *= inv;
}

} // namespace

// ------------------------------------------------------------------ Params

void Params::layout(const Config &cfg)
{
    int o = 0;
    auto take = [&o](int n) { const int r = o; o += n; return r; };
    emb = take(cfg.vocab * D);
    pos = take(T * D);
    ln1g = take(D); ln1b = take(D);
    wq = take(D * D); wk = take(D * D); wv = take(D * D); wo = take(D * D);
    ln2g = take(D); ln2b = take(D);
    w1 = take(D * HID); b1 = take(HID);
    w2 = take(HID * D); b2 = take(D);
    lnfg = take(D); lnfb = take(D);
    wu = take(D * cfg.vocab); bu = take(cfg.vocab);
    total = o;
    w.assign(static_cast<std::size_t>(total), 0.f);
}

void Params::init(const Config &cfg, std::uint32_t seed)
{
    std::mt19937 rng(seed);
    std::normal_distribution<float> gauss(0.f, 1.f);
    auto fill = [&](int off, int n, float scale) {
        for (int i = 0; i < n; ++i) w[static_cast<std::size_t>(off + i)] = gauss(rng) * scale;
    };
    fill(emb, cfg.vocab * D, 0.06f);
    fill(pos, T * D, 0.03f);
    fill(wq, D * D, 0.08f);
    fill(wk, D * D, 0.08f);
    fill(wv, D * D, 0.08f);
    fill(wo, D * D, 0.06f);
    fill(w1, D * HID, 0.08f);
    fill(w2, HID * D, 0.05f);
    fill(wu, D * cfg.vocab, 0.08f);
    for (int i = 0; i < D; ++i)
        w[static_cast<std::size_t>(ln1g + i)] = w[static_cast<std::size_t>(ln2g + i)] = w[static_cast<std::size_t>(lnfg + i)] = 1.f;
}

// ------------------------------------------------------------- Activations

void Activations::allocate(const Config &cfg)
{
    const std::size_t td = T * D, th = T * HID, tv = static_cast<std::size_t>(T) * cfg.vocab;
    for (auto *buf : {&x, &h1, &x1hat, &q, &k, &v, &ctxv, &attn, &a, &h2, &ahat, &m2, &b, &bhat, &hf,
                    &dhf, &db, &da, &dh2, &dctx, &dq, &dk, &dv, &dh1})
        buf->assign(td, 0.f);
    for (auto *buf : {&m1, &gelu, &dgelu, &dm1})
        buf->assign(th, 0.f);
    A.assign(static_cast<std::size_t>(H) * T * T, 0.f);
    probs.assign(tv, 0.f);
    dlogits.assign(tv, 0.f);
    for (auto *buf : {&mu1, &is1, &mu2, &is2, &muf, &isf})
        buf->assign(T, 0.f);
    dArow.assign(T, 0.f);
}

// ------------------------------------------------------------------- Model

void Model::build(std::string_view text)
{
    bool seen[256] = {};
    for (unsigned char c : text) seen[c] = true;
    m_itoc.clear();
    for (int i = 0; i < 256; ++i)
        if (seen[i]) {
            m_ctoi[i] = static_cast<int>(m_itoc.size());
            m_itoc.push_back(static_cast<char>(i));
        }
    cfg.vocab = static_cast<int>(m_itoc.size());
    m_tokens.clear();
    m_tokens.reserve(text.size());
    for (unsigned char c : text) m_tokens.push_back(static_cast<std::uint8_t>(m_ctoi[c]));

    par.layout(cfg);
    par.init(cfg, 0xC0DE);
    adamM.assign(static_cast<std::size_t>(par.total), 0.f);
    adamV.assign(static_cast<std::size_t>(par.total), 0.f);
    step = 0;
}

void Model::forward(const std::uint8_t *seq, Activations &ac) const
{
    const int V = cfg.vocab;
    const float *W = par.w.data();

    // token + position embeddings
    for (int t = 0; t < T; ++t)
        for (int i = 0; i < D; ++i)
            ac.x[t * D + i] = W[par.emb + seq[t] * D + i] + W[par.pos + t * D + i];

    // attention block: a = x + Attn(LN1(x))
    layerNorm(ac.x.data(), W + par.ln1g, W + par.ln1b, ac.h1.data(),
              ac.x1hat.data(), ac.mu1.data(), ac.is1.data());
    matmul(ac.h1.data(), W + par.wq, ac.q.data(), T, D, D);
    matmul(ac.h1.data(), W + par.wk, ac.k.data(), T, D, D);
    matmul(ac.h1.data(), W + par.wv, ac.v.data(), T, D, D);

    const float scale = 1.f / std::sqrt(static_cast<float>(DH));
    std::ranges::fill(ac.ctxv, 0.f);
    for (int h = 0; h < H; ++h) {
        float *A = ac.A.data() + h * T * T;
        for (int t = 0; t < T; ++t) {
            float *row = A + t * T;
            const float *qr = &ac.q[t * D + h * DH];
            for (int s = 0; s <= t; ++s) {
                const float *kr = &ac.k[s * D + h * DH];
                float sc = 0.f;
                for (int i = 0; i < DH; ++i) sc += qr[i] * kr[i];
                row[s] = sc * scale;
            }
            softmaxRow(row, t + 1);               // causal: keys 0..t only
            std::fill(row + t + 1, row + T, 0.f);
            float *cr = &ac.ctxv[t * D + h * DH];
            for (int s = 0; s <= t; ++s) {
                const float a = row[s];
                const float *vr = &ac.v[s * D + h * DH];
                for (int i = 0; i < DH; ++i) cr[i] += a * vr[i];
            }
        }
    }
    matmul(ac.ctxv.data(), W + par.wo, ac.attn.data(), T, D, D);
    for (int i = 0; i < T * D; ++i) ac.a[i] = ac.x[i] + ac.attn[i];

    // MLP block: b = a + W2 gelu(W1 LN2(a) + b1) + b2
    layerNorm(ac.a.data(), W + par.ln2g, W + par.ln2b, ac.h2.data(),
              ac.ahat.data(), ac.mu2.data(), ac.is2.data());
    matmul(ac.h2.data(), W + par.w1, ac.m1.data(), T, D, HID);
    for (int t = 0; t < T; ++t)
        for (int i = 0; i < HID; ++i) {
            ac.m1[t * HID + i] += W[par.b1 + i];
            ac.gelu[t * HID + i] = geluF(ac.m1[t * HID + i]);
        }
    matmul(ac.gelu.data(), W + par.w2, ac.m2.data(), T, HID, D);
    for (int t = 0; t < T; ++t)
        for (int i = 0; i < D; ++i)
            ac.b[t * D + i] = ac.a[t * D + i] + ac.m2[t * D + i] + W[par.b2 + i];

    // unembedding and softmax over the vocabulary
    layerNorm(ac.b.data(), W + par.lnfg, W + par.lnfb, ac.hf.data(),
              ac.bhat.data(), ac.muf.data(), ac.isf.data());
    matmul(ac.hf.data(), W + par.wu, ac.probs.data(), T, D, V);

    for (int t = 0; t < T; ++t) {
        float *lg = &ac.probs[static_cast<std::size_t>(t) * V];
        for (int i = 0; i < V; ++i) lg[i] += W[par.bu + i];
        softmaxRow(lg, V);
    }
}

float Model::loss(const std::uint8_t *seq, const Activations &ac) const
{
    // Every position predicts the token after it; the last one predicts
    // seq[T], the token following the window, so the position the
    // sampler reads is trained like all the others.
    float sum = 0.f;
    for (int t = 0; t < T; ++t)
        sum += -std::log(std::max(ac.probs[static_cast<std::size_t>(t) * cfg.vocab + seq[t + 1]], 1e-9f));
    return sum / T;
}

void Model::backward(const std::uint8_t *seq, Activations &ac, float *grad) const
{
    const int V = cfg.vocab;
    const float *W = par.w.data();
    const float invN = 1.f / T;

    // d loss / d logits = (softmax - onehot) / T
    for (int t = 0; t < T; ++t) {
        const float *p = &ac.probs[static_cast<std::size_t>(t) * V];
        float *dl = &ac.dlogits[static_cast<std::size_t>(t) * V];
        for (int i = 0; i < V; ++i) dl[i] = p[i] * invN;
        dl[seq[t + 1]] -= invN;
    }
    for (int t = 0; t < T; ++t)
        for (int i = 0; i < V; ++i)
            grad[par.bu + i] += ac.dlogits[static_cast<std::size_t>(t) * V + i];

    // logits = hf Wu + bu ; hf = LNf(b)
    std::ranges::fill(ac.dhf, 0.f);
    std::ranges::fill(ac.db, 0.f);
    matmulBack(ac.hf.data(), W + par.wu, ac.dlogits.data(), ac.dhf.data(), grad + par.wu, T, D, V);
    layerNormBack(ac.bhat.data(), W + par.lnfg, ac.isf.data(), ac.dhf.data(), ac.db.data(),
                  grad + par.lnfg, grad + par.lnfb);

    // b = a + gelu(h2 W1 + b1) W2 + b2
    std::ranges::fill(ac.dgelu, 0.f);
    std::ranges::fill(ac.dh2, 0.f);
    for (int t = 0; t < T; ++t)
        for (int i = 0; i < D; ++i) {
            grad[par.b2 + i] += ac.db[t * D + i];
            ac.da[t * D + i] = ac.db[t * D + i]; // residual path
        }
    matmulBack(ac.gelu.data(), W + par.w2, ac.db.data(), ac.dgelu.data(), grad + par.w2, T, HID, D);
    for (int t = 0; t < T; ++t)
        for (int i = 0; i < HID; ++i) {
            ac.dm1[t * HID + i] = ac.dgelu[t * HID + i] * geluD(ac.m1[t * HID + i]);
            grad[par.b1 + i] += ac.dm1[t * HID + i];
        }
    matmulBack(ac.h2.data(), W + par.w1, ac.dm1.data(), ac.dh2.data(), grad + par.w1, T, D, HID);
    layerNormBack(ac.ahat.data(), W + par.ln2g, ac.is2.data(), ac.dh2.data(), ac.da.data(),
                  grad + par.ln2g, grad + par.ln2b);

    // a = x + ctxv Wo : da flows both into the attention output and into x
    std::ranges::fill(ac.dctx, 0.f);
    matmulBack(ac.ctxv.data(), W + par.wo, ac.da.data(), ac.dctx.data(), grad + par.wo, T, D, D);

    // softmax attention backward through the causal mask
    std::ranges::fill(ac.dq, 0.f);
    std::ranges::fill(ac.dk, 0.f);
    std::ranges::fill(ac.dv, 0.f);
    const float scale = 1.f / std::sqrt(static_cast<float>(DH));
    for (int h = 0; h < H; ++h) {
        const float *A = ac.A.data() + h * T * T;
        for (int t = 0; t < T; ++t) {
            const float *row = A + t * T;
            const float *dcr = &ac.dctx[t * D + h * DH];
            float rowdot = 0.f;
            for (int s = 0; s <= t; ++s) {
                const float *vr = &ac.v[s * D + h * DH];
                float dot = 0.f;
                for (int i = 0; i < DH; ++i) dot += dcr[i] * vr[i];
                ac.dArow[s] = dot;
                rowdot += dot * row[s];
            }
            const float *qr = &ac.q[t * D + h * DH];
            float *dqr = &ac.dq[t * D + h * DH];
            for (int s = 0; s <= t; ++s) {
                const float a = row[s];
                const float dS = a * (ac.dArow[s] - rowdot) * scale;
                const float *kr = &ac.k[s * D + h * DH];
                float *dkr = &ac.dk[s * D + h * DH];
                float *dvr = &ac.dv[s * D + h * DH];
                for (int i = 0; i < DH; ++i) {
                    dqr[i] += dS * kr[i];
                    dkr[i] += dS * qr[i];
                    dvr[i] += a * dcr[i];
                }
            }
        }
    }

    std::ranges::fill(ac.dh1, 0.f);
    matmulBack(ac.h1.data(), W + par.wq, ac.dq.data(), ac.dh1.data(), grad + par.wq, T, D, D);
    matmulBack(ac.h1.data(), W + par.wk, ac.dk.data(), ac.dh1.data(), grad + par.wk, T, D, D);
    matmulBack(ac.h1.data(), W + par.wv, ac.dv.data(), ac.dh1.data(), grad + par.wv, T, D, D);
    // dx = da (residual) + LN1 backward; da is no longer needed afterwards
    layerNormBack(ac.x1hat.data(), W + par.ln1g, ac.is1.data(), ac.dh1.data(), ac.da.data(),
                  grad + par.ln1g, grad + par.ln1b);

    for (int t = 0; t < T; ++t)
        for (int i = 0; i < D; ++i) {
            grad[par.emb + seq[t] * D + i] += ac.da[t * D + i];
            grad[par.pos + t * D + i] += ac.da[t * D + i];
        }
}

void Model::adam(const float *grad, float lr)
{
    ++step;
    constexpr float beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f;
    const float c1 = 1.f - std::pow(beta1, static_cast<float>(step));
    const float c2 = 1.f - std::pow(beta2, static_cast<float>(step));
    for (int i = 0; i < par.total; ++i) {
        const float g = grad[i];
        adamM[i] = beta1 * adamM[i] + (1.f - beta1) * g;
        adamV[i] = beta2 * adamV[i] + (1.f - beta2) * g * g;
        par.w[i] -= lr * (adamM[i] / c1) / (std::sqrt(adamV[i] / c2) + eps);
    }
}

std::string Model::generate(int nChars, float temperature, std::uint32_t seed) const
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> uniform(0.f, 1.f);
    std::vector<std::uint8_t> window(T, 0);
    // prime the window with a random corpus slice
    if (m_tokens.size() > static_cast<std::size_t>(T) + 1) {
        const auto start = static_cast<std::size_t>(uniform(rng) * static_cast<float>(m_tokens.size() - T - 1));
        std::copy_n(m_tokens.begin() + static_cast<std::ptrdiff_t>(start), T, window.begin());
    }
    Activations ac;
    ac.allocate(cfg);
    std::vector<float> p(static_cast<std::size_t>(cfg.vocab));
    std::string out;
    out.reserve(static_cast<std::size_t>(nChars));
    for (int n = 0; n < nChars; ++n) {
        forward(window.data(), ac);
        const float *last = &ac.probs[static_cast<std::size_t>(T - 1) * cfg.vocab];
        std::copy_n(last, cfg.vocab, p.begin());
        if (temperature != 1.f) {
            // p^(1/temp) renormalized == softmax(logits / temp)
            float sum = 0.f;
            for (float &x : p) { x = std::pow(x, 1.f / temperature); sum += x; }
            for (float &x : p) x /= sum;
        }
        const float r = uniform(rng);
        float acc = 0.f;
        int pick = cfg.vocab - 1;
        for (int i = 0; i < cfg.vocab; ++i) {
            acc += p[static_cast<std::size_t>(i)];
            if (r <= acc) { pick = i; break; }
        }
        out += itoc(pick);
        std::ranges::rotate(window, window.begin() + 1);
        window.back() = static_cast<std::uint8_t>(pick);
    }
    return out;
}

} // namespace gpt
