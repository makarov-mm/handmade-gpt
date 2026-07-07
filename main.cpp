// HandmadeGPT — a tiny character-level GPT trained live, from scratch.
// One pre-LN transformer block: embeddings + learned positions, 2-head causal
// self-attention, GELU MLP, layer norms, untied unembedding. Forward, backward
// and Adam are all hand-written — no ML libraries, no autograd, just C++.
// Live visualization: loss curve, per-head attention maps, PCA of the
// character embeddings, and text sampled from the model as it learns.
// C++17 / Qt6 Widgets, no other dependencies.

#include <QApplication>
#include <QWidget>
#include <QPainter>
#include <QTimer>
#include <QElapsedTimer>
#include <QFileDialog>
#include <QHBoxLayout>
#include <QFormLayout>
#include <QDoubleSpinBox>
#include <QSpinBox>
#include <QPushButton>
#include <QLabel>
#include <QGroupBox>
#include <QStyleFactory>

#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <fstream>
#include <random>
#include <thread>
#include <vector>

static inline float clampf(float v, float lo, float hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}

// Built-in training corpus (original text; deliberately plain and
// repetitive — a char model picks up the patterns quickly).
static const char *kCorpus =
"the river rises in the hills and the river runs down to the sea. the rain "
"falls on the slopes and the water gathers in the valley. the wind moves "
"over the water and the waves follow the wind across the lake. the sun "
"climbs in the morning and the sun settles behind the ridge in the evening. "
"the light fades in the west and the stars come out over the fields. "
"the birds wake before the light and the birds sing until the heat of the "
"day. the leaves turn in the autumn and the leaves fall when the frost "
"comes. the snow covers the ground in winter and the snow melts into the "
"streams in spring. the seeds wait in the soil and the seeds open when the "
"ground warms. the roots reach down for the water and the branches reach up "
"for the light. the moon pulls the tide up the shore and the tide slides "
"back down the sand. the fog settles in the low ground at night and the fog "
"lifts when the morning warms the air. the path follows the stream through "
"the woods and the path climbs out of the trees onto the open hill. the "
"stone holds the heat of the day and the stone gives the heat back in the "
"dark. the fire needs the air and the fire eats the wood down to the ash. "
"the bread rises in the warmth and the bread browns in the heat of the "
"oven. the letters make the words and the words make the lines and the "
"lines fill the page. the reader follows the lines and the story unfolds "
"one word at a time. the old man mends the net in the morning and the old "
"man sails with the tide in the afternoon. the harvest comes at the end of "
"the summer and the fields rest under the snow until the spring returns. ";

// ------------------------------------------------------------------- model

struct Config {
    int V = 0;          // vocab (from corpus)
    int d = 64;         // model width
    int H = 2;          // heads
    int ctx = 64;       // context length
    int hid = 256;      // MLP hidden
};

struct Params {
    std::vector<float> w;   // all parameters in one flat vector
    // offsets
    int emb, pos, ln1g, ln1b, wq, wk, wv, wo, ln2g, ln2b,
        w1, b1, w2, b2, lnfg, lnfb, wu, bu, total;

    void layout(const Config &c)
    {
        int o = 0;
        auto take = [&o](int n) { int r = o; o += n; return r; };
        emb  = take(c.V * c.d);
        pos  = take(c.ctx * c.d);
        ln1g = take(c.d); ln1b = take(c.d);
        wq = take(c.d * c.d); wk = take(c.d * c.d);
        wv = take(c.d * c.d); wo = take(c.d * c.d);
        ln2g = take(c.d); ln2b = take(c.d);
        w1 = take(c.d * c.hid); b1 = take(c.hid);
        w2 = take(c.hid * c.d); b2 = take(c.d);
        lnfg = take(c.d); lnfb = take(c.d);
        wu = take(c.d * c.V); bu = take(c.V);
        total = o;
        w.assign(total, 0.f);
    }

    void init(const Config &c, uint32_t seed)
    {
        std::mt19937 rng(seed);
        std::normal_distribution<float> g(0.f, 1.f);
        auto fill = [&](int off, int n, float s) {
            for (int i = 0; i < n; ++i) w[off + i] = g(rng) * s;
        };
        fill(emb, c.V * c.d, 0.06f);
        fill(pos, c.ctx * c.d, 0.03f);
        fill(wq, c.d * c.d, 0.08f); fill(wk, c.d * c.d, 0.08f);
        fill(wv, c.d * c.d, 0.08f); fill(wo, c.d * c.d, 0.06f);
        fill(w1, c.d * c.hid, 0.08f);
        fill(w2, c.hid * c.d, 0.05f);
        fill(wu, c.d * c.V, 0.08f);
        for (int i = 0; i < c.d; ++i) {
            w[ln1g + i] = w[ln2g + i] = w[lnfg + i] = 1.f;
        }
    }
};

// activations for one sequence (kept for backprop)
struct Acts {
    std::vector<float> x, h1, x1hat, q, k, v, A, ctxv, attn, a,
                       h2, ahat, m1, gelu, b, bhat, probs;
    std::vector<float> mu1, is1, mu2, is2, muf, isf; // LN stats per position
    void alloc(const Config &c)
    {
        const int T = c.ctx, d = c.d;
        x.resize(T * d); h1.resize(T * d); x1hat.resize(T * d);
        q.resize(T * d); k.resize(T * d); v.resize(T * d);
        A.resize(c.H * T * T);
        ctxv.resize(T * d); attn.resize(T * d); a.resize(T * d);
        h2.resize(T * d); ahat.resize(T * d);
        m1.resize(T * c.hid); gelu.resize(T * c.hid);
        b.resize(T * d); bhat.resize(T * d);
        probs.resize(T * c.V);
        mu1.resize(T); is1.resize(T); mu2.resize(T); is2.resize(T);
        muf.resize(T); isf.resize(T);
    }
};

static inline float geluF(float x)
{
    const float u = 0.7978845608f * (x + 0.044715f * x * x * x);
    return 0.5f * x * (1.f + std::tanh(u));
}
static inline float geluD(float x)
{
    const float u = 0.7978845608f * (x + 0.044715f * x * x * x);
    const float t = std::tanh(u);
    const float du = 0.7978845608f * (1.f + 3.f * 0.044715f * x * x);
    return 0.5f * (1.f + t) + 0.5f * x * (1.f - t * t) * du;
}

// y = x @ W  (x: [T,in], W: [in,out], y: [T,out])
static void matmul(const float *x, const float *W, float *y,
                   int T, int in, int out)
{
    for (int t = 0; t < T; ++t) {
        const float *xr = x + t * in;
        float *yr = y + t * out;
        std::memset(yr, 0, sizeof(float) * out);
        for (int i = 0; i < in; ++i) {
            const float xi = xr[i];
            const float *wr = W + i * out;
            for (int o = 0; o < out; ++o) yr[o] += xi * wr[o];
        }
    }
}
// dX += dY @ W^T ; dW += X^T @ dY
static void matmulBack(const float *x, const float *W, const float *dy,
                       float *dx, float *dW, int T, int in, int out)
{
    for (int t = 0; t < T; ++t) {
        const float *xr = x + t * in;
        const float *dyr = dy + t * out;
        float *dxr = dx + t * in;
        for (int i = 0; i < in; ++i) {
            const float *wr = W + i * out;
            float *dwr = dW + i * out;
            float acc = 0.f;
            const float xi = xr[i];
            for (int o = 0; o < out; ++o) {
                acc += dyr[o] * wr[o];
                dwr[o] += xi * dyr[o];
            }
            dxr[i] += acc;
        }
    }
}

static void layerNorm(const float *x, const float *g, const float *b,
                      float *y, float *xhat, float *muO, float *isO,
                      int T, int d)
{
    for (int t = 0; t < T; ++t) {
        const float *xr = x + t * d;
        float mu = 0.f;
        for (int i = 0; i < d; ++i) mu += xr[i];
        mu /= d;
        float var = 0.f;
        for (int i = 0; i < d; ++i) { float c = xr[i] - mu; var += c * c; }
        const float is = 1.f / std::sqrt(var / d + 1e-5f);
        muO[t] = mu; isO[t] = is;
        for (int i = 0; i < d; ++i) {
            const float xh = (xr[i] - mu) * is;
            xhat[t * d + i] = xh;
            y[t * d + i] = xh * g[i] + b[i];
        }
    }
}
static void layerNormBack(const float *xhat, const float *g, const float *is,
                          const float *dy, float *dx, float *dg, float *db,
                          int T, int d)
{
    for (int t = 0; t < T; ++t) {
        const float *xh = xhat + t * d;
        const float *dyr = dy + t * d;
        float s1 = 0.f, s2 = 0.f;
        for (int i = 0; i < d; ++i) {
            const float dyg = dyr[i] * g[i];
            s1 += dyg;
            s2 += dyg * xh[i];
            dg[i] += dyr[i] * xh[i];
            db[i] += dyr[i];
        }
        s1 /= d; s2 /= d;
        float *dxr = dx + t * d;
        for (int i = 0; i < d; ++i)
            dxr[i] += (dyr[i] * g[i] - s1 - xh[i] * s2) * is[t];
    }
}

class Model
{
public:
    Config cfg;
    Params par;
    std::vector<float> adamM, adamV;
    int step = 0;

    void build(const std::string &text)
    {
        // vocab from corpus
        bool seen[256] = { false };
        for (unsigned char c : text) seen[c] = true;
        m_itoc.clear();
        for (int i = 0; i < 256; ++i)
            if (seen[i]) { m_ctoi[i] = int(m_itoc.size()); m_itoc.push_back(char(i)); }
        cfg.V = int(m_itoc.size());
        m_tokens.clear();
        m_tokens.reserve(text.size());
        for (unsigned char c : text) m_tokens.push_back(uint8_t(m_ctoi[c]));

        par.layout(cfg);
        par.init(cfg, 0xC0DE);
        adamM.assign(par.total, 0.f);
        adamV.assign(par.total, 0.f);
        step = 0;
    }

    int vocab() const { return cfg.V; }
    int paramCount() const { return par.total; }
    const std::vector<uint8_t> &tokens() const { return m_tokens; }
    char itoc(int i) const { return m_itoc[i]; }

    // forward one sequence; returns mean loss; fills acts
    float forward(const uint8_t *seq, Acts &ac) const
    {
        const int T = cfg.ctx, d = cfg.d, H = cfg.H, dh = d / H,
                  hid = cfg.hid, V = cfg.V;
        const float *W = par.w.data();

        for (int t = 0; t < T; ++t)
            for (int i = 0; i < d; ++i)
                ac.x[t * d + i] = W[par.emb + seq[t] * d + i]
                                + W[par.pos + t * d + i];

        layerNorm(ac.x.data(), W + par.ln1g, W + par.ln1b, ac.h1.data(),
                  ac.x1hat.data(), ac.mu1.data(), ac.is1.data(), T, d);
        matmul(ac.h1.data(), W + par.wq, ac.q.data(), T, d, d);
        matmul(ac.h1.data(), W + par.wk, ac.k.data(), T, d, d);
        matmul(ac.h1.data(), W + par.wv, ac.v.data(), T, d, d);

        const float scale = 1.f / std::sqrt(float(dh));
        std::fill(ac.ctxv.begin(), ac.ctxv.end(), 0.f);
        for (int h = 0; h < H; ++h) {
            float *A = ac.A.data() + h * T * T;
            for (int t = 0; t < T; ++t) {
                float mx = -1e30f;
                for (int s = 0; s <= t; ++s) {
                    float sc = 0.f;
                    const float *qr = &ac.q[t * d + h * dh];
                    const float *kr = &ac.k[s * d + h * dh];
                    for (int i = 0; i < dh; ++i) sc += qr[i] * kr[i];
                    sc *= scale;
                    A[t * T + s] = sc;
                    mx = std::max(mx, sc);
                }
                float sum = 0.f;
                for (int s = 0; s <= t; ++s) {
                    A[t * T + s] = std::exp(A[t * T + s] - mx);
                    sum += A[t * T + s];
                }
                const float inv = 1.f / sum;
                for (int s = 0; s <= t; ++s) A[t * T + s] *= inv;
                for (int s = t + 1; s < T; ++s) A[t * T + s] = 0.f;
                float *cr = &ac.ctxv[t * d + h * dh];
                for (int s = 0; s <= t; ++s) {
                    const float a = A[t * T + s];
                    const float *vr = &ac.v[s * d + h * dh];
                    for (int i = 0; i < dh; ++i) cr[i] += a * vr[i];
                }
            }
        }
        matmul(ac.ctxv.data(), W + par.wo, ac.attn.data(), T, d, d);
        for (int i = 0; i < T * d; ++i) ac.a[i] = ac.x[i] + ac.attn[i];

        layerNorm(ac.a.data(), W + par.ln2g, W + par.ln2b, ac.h2.data(),
                  ac.ahat.data(), ac.mu2.data(), ac.is2.data(), T, d);
        matmul(ac.h2.data(), W + par.w1, ac.m1.data(), T, d, hid);
        for (int t = 0; t < T; ++t)
            for (int i = 0; i < hid; ++i) {
                ac.m1[t * hid + i] += W[par.b1 + i];
                ac.gelu[t * hid + i] = geluF(ac.m1[t * hid + i]);
            }
        std::vector<float> m2(T * d, 0.f);
        matmul(ac.gelu.data(), W + par.w2, m2.data(), T, hid, d);
        for (int t = 0; t < T; ++t)
            for (int i = 0; i < d; ++i)
                ac.b[t * d + i] = ac.a[t * d + i] + m2[t * d + i]
                                + W[par.b2 + i];

        std::vector<float> hf(T * d);
        layerNorm(ac.b.data(), W + par.lnfg, W + par.lnfb, hf.data(),
                  ac.bhat.data(), ac.muf.data(), ac.isf.data(), T, d);
        matmul(hf.data(), W + par.wu, ac.probs.data(), T, d, V);

        float loss = 0.f;
        for (int t = 0; t < T; ++t) {
            float *lg = &ac.probs[t * V];
            float mx = -1e30f;
            for (int i = 0; i < V; ++i) { lg[i] += W[par.bu + i]; mx = std::max(mx, lg[i]); }
            float sum = 0.f;
            for (int i = 0; i < V; ++i) { lg[i] = std::exp(lg[i] - mx); sum += lg[i]; }
            const float inv = 1.f / sum;
            for (int i = 0; i < V; ++i) lg[i] *= inv;
            if (t < T - 1)
                loss += -std::log(std::max(lg[seq[t + 1]], 1e-9f));
        }
        return loss / (T - 1);
    }

    // backward one sequence into grad buffer g (same layout as params)
    void backward(const uint8_t *seq, const Acts &ac, float *gr) const
    {
        const int T = cfg.ctx, d = cfg.d, H = cfg.H, dh = d / H,
                  hid = cfg.hid, V = cfg.V;
        const float *W = par.w.data();
        const float invN = 1.f / (T - 1);

        std::vector<float> dlog(T * V, 0.f);
        for (int t = 0; t < T - 1; ++t) {
            const float *p = &ac.probs[t * V];
            float *dl = &dlog[t * V];
            for (int i = 0; i < V; ++i) dl[i] = p[i] * invN;
            dl[seq[t + 1]] -= invN;
        }
        for (int t = 0; t < T; ++t)
            for (int i = 0; i < V; ++i)
                gr[par.bu + i] += dlog[t * V + i];

        // hf = LNf(b); logits = hf Wu — reconstruct hf from bhat
        std::vector<float> hf(T * d), dhf(T * d, 0.f), db(T * d, 0.f);
        for (int t = 0; t < T; ++t)
            for (int i = 0; i < d; ++i)
                hf[t * d + i] = ac.bhat[t * d + i] * W[par.lnfg + i]
                              + W[par.lnfb + i];
        matmulBack(hf.data(), W + par.wu, dlog.data(), dhf.data(),
                   gr + par.wu, T, d, V);
        layerNormBack(ac.bhat.data(), W + par.lnfg, ac.isf.data(),
                      dhf.data(), db.data(), gr + par.lnfg, gr + par.lnfb,
                      T, d);

        // b = a + gelu(h2 W1 + b1) W2 + b2
        std::vector<float> dgelu(T * hid, 0.f), da(T * d, 0.f),
                           dh2(T * d, 0.f), dm1(T * hid, 0.f);
        for (int t = 0; t < T; ++t)
            for (int i = 0; i < d; ++i) {
                gr[par.b2 + i] += db[t * d + i];
                da[t * d + i] += db[t * d + i]; // residual
            }
        matmulBack(ac.gelu.data(), W + par.w2, db.data(), dgelu.data(),
                   gr + par.w2, T, hid, d);
        for (int t = 0; t < T; ++t)
            for (int i = 0; i < hid; ++i) {
                dm1[t * hid + i] = dgelu[t * hid + i]
                                 * geluD(ac.m1[t * hid + i]);
                gr[par.b1 + i] += dm1[t * hid + i];
            }
        matmulBack(ac.h2.data(), W + par.w1, dm1.data(), dh2.data(),
                   gr + par.w1, T, d, hid);
        layerNormBack(ac.ahat.data(), W + par.ln2g, ac.is2.data(),
                      dh2.data(), da.data(), gr + par.ln2g, gr + par.ln2b,
                      T, d);

        // a = x + ctxv Wo
        std::vector<float> dattn = da; // residual splits
        std::vector<float> dx = da;
        std::vector<float> dctx(T * d, 0.f);
        matmulBack(ac.ctxv.data(), W + par.wo, dattn.data(), dctx.data(),
                   gr + par.wo, T, d, d);

        std::vector<float> dq(T * d, 0.f), dk(T * d, 0.f), dv(T * d, 0.f);
        const float scale = 1.f / std::sqrt(float(dh));
        for (int h = 0; h < H; ++h) {
            const float *A = ac.A.data() + h * T * T;
            for (int t = 0; t < T; ++t) {
                // dA and softmax backward
                float rowdot = 0.f;
                std::vector<float> dArow(t + 1);
                for (int s = 0; s <= t; ++s) {
                    float dot = 0.f;
                    const float *dcr = &dctx[t * d + h * dh];
                    const float *vr = &ac.v[s * d + h * dh];
                    for (int i = 0; i < dh; ++i) dot += dcr[i] * vr[i];
                    dArow[s] = dot;
                    rowdot += dot * A[t * T + s];
                }
                for (int s = 0; s <= t; ++s) {
                    const float a = A[t * T + s];
                    const float dS = a * (dArow[s] - rowdot) * scale;
                    float *dqr = &dq[t * d + h * dh];
                    float *dkr = &dk[s * d + h * dh];
                    const float *qr = &ac.q[t * d + h * dh];
                    const float *kr = &ac.k[s * d + h * dh];
                    float *dvr = &dv[s * d + h * dh];
                    const float *dcr = &dctx[t * d + h * dh];
                    for (int i = 0; i < dh; ++i) {
                        dqr[i] += dS * kr[i];
                        dkr[i] += dS * qr[i];
                        dvr[i] += a * dcr[i];
                    }
                }
            }
        }

        std::vector<float> dh1(T * d, 0.f);
        matmulBack(ac.h1.data(), W + par.wq, dq.data(), dh1.data(),
                   gr + par.wq, T, d, d);
        matmulBack(ac.h1.data(), W + par.wk, dk.data(), dh1.data(),
                   gr + par.wk, T, d, d);
        matmulBack(ac.h1.data(), W + par.wv, dv.data(), dh1.data(),
                   gr + par.wv, T, d, d);
        layerNormBack(ac.x1hat.data(), W + par.ln1g, ac.is1.data(),
                      dh1.data(), dx.data(), gr + par.ln1g, gr + par.ln1b,
                      T, d);

        for (int t = 0; t < T; ++t)
            for (int i = 0; i < d; ++i) {
                gr[par.emb + seq[t] * d + i] += dx[t * d + i];
                gr[par.pos + t * d + i] += dx[t * d + i];
            }
    }

    void adam(const float *grad, float lr)
    {
        ++step;
        const float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
        const float c1 = 1.f - std::pow(b1, float(step));
        const float c2 = 1.f - std::pow(b2, float(step));
        for (int i = 0; i < par.total; ++i) {
            adamM[i] = b1 * adamM[i] + (1.f - b1) * grad[i];
            adamV[i] = b2 * adamV[i] + (1.f - b2) * grad[i] * grad[i];
            par.w[i] -= lr * (adamM[i] / c1)
                        / (std::sqrt(adamV[i] / c2) + eps);
        }
    }

    // sample text from the model
    std::string generate(int nChars, float temperature, uint32_t seed) const
    {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> u(0.f, 1.f);
        std::vector<uint8_t> win(cfg.ctx, 0);
        // seed the window with a random corpus slice
        if (int(m_tokens.size()) > cfg.ctx + 1) {
            int st = int(u(rng) * (m_tokens.size() - cfg.ctx - 1));
            std::copy(m_tokens.begin() + st, m_tokens.begin() + st + cfg.ctx,
                      win.begin());
        }
        Acts ac; ac.alloc(cfg);
        std::string out;
        for (int n = 0; n < nChars; ++n) {
            forward(win.data(), ac);
            const float *p0 = &ac.probs[(cfg.ctx - 1) * cfg.V];
            std::vector<float> p(p0, p0 + cfg.V);
            if (temperature != 1.f) {
                float sum = 0.f;
                for (auto &x : p) { x = std::pow(x, 1.f / temperature); sum += x; }
                for (auto &x : p) x /= sum;
            }
            float r = u(rng), acc = 0.f;
            int pick = cfg.V - 1;
            for (int i = 0; i < cfg.V; ++i) {
                acc += p[i];
                if (r <= acc) { pick = i; break; }
            }
            out += m_itoc[pick];
            std::rotate(win.begin(), win.begin() + 1, win.end());
            win.back() = uint8_t(pick);
        }
        return out;
    }

private:
    std::vector<char> m_itoc;
    int m_ctoi[256] = { 0 };
    std::vector<uint8_t> m_tokens;
};

// ------------------------------------------------------------------ trainer

class Trainer
{
public:
    Model model;
    float lr = 3e-3f;
    int batch = 12;
    float lastLoss = 0.f;
    std::vector<float> lossHistory;
    std::vector<float> lastAttn; // copy of head attention maps for viz
    float tokensPerSec = 0.f;

    void build(const std::string &text) { model.build(text); lossHistory.clear(); }

    void trainSteps(int nSteps)
    {
        const int T = std::max(2u, std::thread::hardware_concurrency());
        const int P = model.paramCount();
        std::vector<std::vector<float>> grads(T);
        std::vector<std::vector<Acts>> acts(T);
        static std::mt19937 rng(0xF17E55);
        QElapsedTimer clk; clk.start();
        int tokensDone = 0;

        for (int s = 0; s < nSteps; ++s) {
            const auto &tok = model.tokens();
            std::uniform_int_distribution<int>
                starts(0, int(tok.size()) - model.cfg.ctx - 2);
            std::vector<int> off(batch);
            for (auto &o : off) o = starts(rng);

            std::vector<float> losses(batch, 0.f);
            std::vector<std::thread> th;
            for (int t = 0; t < T; ++t) {
                th.emplace_back([&, t] {
                    if (grads[t].size() != size_t(P))
                        grads[t].assign(P, 0.f);
                    else
                        std::fill(grads[t].begin(), grads[t].end(), 0.f);
                    if (acts[t].empty()) { acts[t].resize(1); acts[t][0].alloc(model.cfg); }
                    for (int b = t; b < batch; b += T) {
                        const uint8_t *seq = tok.data() + off[b];
                        losses[b] = model.forward(seq, acts[t][0]);
                        model.backward(seq, acts[t][0], grads[t].data());
                        if (b == 0)
                            lastAttn.assign(acts[t][0].A.begin(),
                                            acts[t][0].A.end());
                    }
                });
            }
            for (auto &x : th) x.join();

            // reduce grads, mean over batch
            float *g0 = grads[0].data();
            for (int t = 1; t < T; ++t) {
                const float *gt = grads[t].data();
                for (int i = 0; i < P; ++i) g0[i] += gt[i];
            }
            const float invB = 1.f / batch;
            for (int i = 0; i < P; ++i) g0[i] *= invB;

            model.adam(g0, lr);

            float L = 0.f;
            for (float l : losses) L += l;
            lastLoss = L / batch;
            lossHistory.push_back(lastLoss);
            if (lossHistory.size() > 4000)
                lossHistory.erase(lossHistory.begin(),
                                  lossHistory.begin() + 2000);
            tokensDone += batch * model.cfg.ctx;
        }
        tokensPerSec = tokensDone / std::max(1e-3f,
                        float(clk.nsecsElapsed()) * 1e-9f);
    }
};

// -------------------------------------------------------------------- viz

class TransformerCanvas : public QWidget
{
    Q_OBJECT
public:
    Trainer *trainer = nullptr;
    QString sample = "…";

    explicit TransformerCanvas(QWidget *parent = nullptr) : QWidget(parent)
    {
        setMinimumSize(700, 560);
    }

protected:
    void paintEvent(QPaintEvent *) override
    {
        QPainter p(this);
        p.fillRect(rect(), QColor(12, 12, 18));
        if (!trainer) return;
        const int W2 = width(), H2 = height();
        const int mid = H2 * 55 / 100;

        drawLoss(p, QRect(8, 8, W2 / 2 - 12, mid - 16));
        drawAttention(p, QRect(W2 / 2 + 4, 8, W2 / 2 - 12, mid - 16));
        drawPCA(p, QRect(8, mid + 4, W2 / 2 - 12, H2 - mid - 12));
        drawText(p, QRect(W2 / 2 + 4, mid + 4, W2 / 2 - 12, H2 - mid - 12));
    }

private:
    void frameRect(QPainter &p, const QRect &r, const QString &title)
    {
        p.setPen(QColor(70, 75, 90));
        p.setBrush(QColor(18, 18, 26));
        p.drawRect(r);
        p.setPen(QColor(150, 160, 180));
        p.drawText(r.adjusted(8, 4, -8, 0), Qt::AlignTop | Qt::AlignLeft, title);
    }

    void drawLoss(QPainter &p, const QRect &r)
    {
        frameRect(p, r, "loss");
        const auto &h = trainer->lossHistory;
        if (h.size() < 2) return;
        float mx = 0.f, mn = 1e9f;
        for (float v : h) { mx = std::max(mx, v); mn = std::min(mn, v); }
        mn = std::min(mn, 0.5f);
        const QRect in = r.adjusted(8, 22, -8, -8);
        p.setPen(QPen(QColor(120, 230, 160), 1.5));
        QPointF prev;
        for (size_t i = 0; i < h.size(); ++i) {
            const float fx = in.x() + in.width() * float(i) / (h.size() - 1);
            const float fy = in.bottom()
                - in.height() * (h[i] - mn) / std::max(0.1f, mx - mn);
            if (i) p.drawLine(prev, QPointF(fx, fy));
            prev = { fx, fy };
        }
        p.setPen(QColor(160, 170, 190));
        p.drawText(in, Qt::AlignTop | Qt::AlignRight,
                   QString("%1").arg(trainer->lastLoss, 0, 'f', 3));
    }

    void drawAttention(QPainter &p, const QRect &r)
    {
        frameRect(p, r, "attention  head 0 / head 1");
        const auto &A = trainer->lastAttn;
        const int T = trainer->model.cfg.ctx, H = trainer->model.cfg.H;
        if (int(A.size()) < H * T * T) return;
        const QRect in = r.adjusted(8, 22, -8, -8);
        const int cell = std::max(1, std::min(in.width() / (H * T + 4),
                                              in.height() / T));
        for (int h = 0; h < H; ++h) {
            const int ox = in.x() + h * (cell * T + 8);
            QImage img(T, T, QImage::Format_RGB32);
            for (int t = 0; t < T; ++t) {
                QRgb *row = reinterpret_cast<QRgb *>(img.scanLine(t));
                for (int s = 0; s < T; ++s) {
                    const float a = clampf(A[h * T * T + t * T + s] * 3.f,
                                           0.f, 1.f);
                    row[s] = h == 0
                        ? qRgb(int(30 + 200 * a), int(30 + 120 * a), 40)
                        : qRgb(40, int(30 + 140 * a), int(30 + 210 * a));
                }
            }
            p.drawImage(QRect(ox, in.y(), cell * T, cell * T), img);
        }
    }

    void drawPCA(QPainter &p, const QRect &r)
    {
        frameRect(p, r, "embedding PCA");
        const Model &m = trainer->model;
        const int V = m.cfg.V, d = m.cfg.d;
        const float *E = m.par.w.data() + m.par.emb;
        // mean-center
        std::vector<float> mean(d, 0.f);
        for (int v = 0; v < V; ++v)
            for (int i = 0; i < d; ++i) mean[i] += E[v * d + i];
        for (auto &x : mean) x /= V;
        // top-2 principal directions via power iteration on covariance
        auto power = [&](std::vector<float> &dir, const std::vector<float> *orth) {
            std::mt19937 rng(7 + (orth != nullptr));
            std::normal_distribution<float> g;
            dir.resize(d);
            for (auto &x : dir) x = g(rng);
            std::vector<float> nd(d);
            for (int it = 0; it < 12; ++it) {
                std::fill(nd.begin(), nd.end(), 0.f);
                for (int v = 0; v < V; ++v) {
                    float dot = 0.f;
                    for (int i = 0; i < d; ++i)
                        dot += (E[v * d + i] - mean[i]) * dir[i];
                    for (int i = 0; i < d; ++i)
                        nd[i] += dot * (E[v * d + i] - mean[i]);
                }
                if (orth) {
                    float dot = 0.f;
                    for (int i = 0; i < d; ++i) dot += nd[i] * (*orth)[i];
                    for (int i = 0; i < d; ++i) nd[i] -= dot * (*orth)[i];
                }
                float nrm = 1e-9f;
                for (float x : nd) nrm += x * x;
                nrm = 1.f / std::sqrt(nrm);
                for (int i = 0; i < d; ++i) dir[i] = nd[i] * nrm;
            }
        };
        std::vector<float> p1, p2;
        power(p1, nullptr);
        power(p2, &p1);

        std::vector<float> px(V), py(V);
        float mx = 1e-6f;
        for (int v = 0; v < V; ++v) {
            float a = 0.f, b = 0.f;
            for (int i = 0; i < d; ++i) {
                a += (E[v * d + i] - mean[i]) * p1[i];
                b += (E[v * d + i] - mean[i]) * p2[i];
            }
            px[v] = a; py[v] = b;
            mx = std::max({ mx, std::fabs(a), std::fabs(b) });
        }
        const QRect in = r.adjusted(10, 24, -10, -10);
        QFont f = p.font(); f.setPointSize(9); f.setBold(true); p.setFont(f);
        for (int v = 0; v < V; ++v) {
            const float fx = in.center().x() + px[v] / mx * in.width() * 0.46f;
            const float fy = in.center().y() - py[v] / mx * in.height() * 0.46f;
            const char c = m.itoc(v);
            const bool vowel = strchr("aeiou", c) != nullptr;
            p.setPen(c == ' ' ? QColor(240, 220, 100)
                   : vowel   ? QColor(120, 220, 160)
                             : QColor(140, 170, 240));
            p.drawText(QPointF(fx, fy), c == ' ' ? QString("_") : QString(c));
        }
    }

    void drawText(QPainter &p, const QRect &r)
    {
        frameRect(p, r, "sampled text");
        p.setPen(QColor(210, 215, 225));
        QFont f("monospace"); f.setPointSize(9); p.setFont(f);
        p.drawText(r.adjusted(10, 26, -10, -8),
                   Qt::TextWordWrap | Qt::AlignTop, sample);
    }
};

// -------------------------------------------------------------------- window

class MainWindow : public QWidget
{
    Q_OBJECT
public:
    MainWindow()
    {
        setWindowTitle("HandmadeGPT — a transformer with no libraries");
        m_trainer.build(kCorpus);

        auto *canvas = new TransformerCanvas;
        canvas->trainer = &m_trainer;

        auto *lr = new QDoubleSpinBox;
        lr->setRange(1e-4, 2e-2); lr->setDecimals(4); lr->setSingleStep(5e-4);
        lr->setValue(m_trainer.lr);

        auto *batch = new QSpinBox;
        batch->setRange(2, 48);
        batch->setValue(m_trainer.batch);

        auto *stepsPerFrame = new QSpinBox;
        stepsPerFrame->setRange(0, 20);
        stepsPerFrame->setValue(2);

        auto *temp = new QDoubleSpinBox;
        temp->setRange(0.2, 2.0); temp->setSingleStep(0.1);
        temp->setValue(0.8);

        auto *gen = new QPushButton("Generate now");
        auto *resetM = new QPushButton("Reset model");
        auto *load = new QPushButton("Load text file…");

        auto *stats = new QLabel;
        stats->setStyleSheet("color:#8fa;font-family:monospace");

        auto *form = new QFormLayout;
        form->addRow("Learning rate", lr);
        form->addRow("Batch", batch);
        form->addRow("Steps/frame", stepsPerFrame);
        form->addRow("Temperature", temp);
        form->addRow(gen);
        form->addRow(resetM);
        form->addRow(load);
        form->addRow(stats);

        auto *group = new QGroupBox("Parameters");
        group->setLayout(form);
        group->setFixedWidth(280);

        auto *layout = new QHBoxLayout(this);
        layout->addWidget(group);
        layout->addWidget(canvas, 1);

        connect(lr, &QDoubleSpinBox::valueChanged, this, [this](double v) { m_trainer.lr = float(v); });
        connect(batch, &QSpinBox::valueChanged, this, [this](int v) { m_trainer.batch = v; });
        connect(temp, &QDoubleSpinBox::valueChanged, this, [this](double v) { m_temp = float(v); });
        connect(gen, &QPushButton::clicked, this, [this, canvas] {
            canvas->sample = QString::fromStdString(
                m_trainer.model.generate(360, m_temp, m_genSeed++));
            canvas->update();
        });
        connect(resetM, &QPushButton::clicked, this, [this, canvas] {
            m_trainer.build(m_text);
            canvas->sample = "…";
        });
        connect(load, &QPushButton::clicked, this, [this, canvas] {
            const QString fn = QFileDialog::getOpenFileName(
                this, "Training text", {}, "Text files (*.txt);;All files (*)");
            if (fn.isEmpty()) return;
            std::ifstream f(fn.toStdString(), std::ios::binary);
            std::string s((std::istreambuf_iterator<char>(f)),
                          std::istreambuf_iterator<char>());
            if (s.size() > 200000) s.resize(200000);
            if (int(s.size()) > m_trainer.model.cfg.ctx * 4) {
                m_text = s;
                m_trainer.build(m_text);
                canvas->sample = "…";
            }
        });

        connect(&m_timer, &QTimer::timeout, this, [this, canvas, stepsPerFrame, stats] {
            const int n = stepsPerFrame->value();
            if (n > 0) m_trainer.trainSteps(n);
            if (m_trainer.model.step - m_lastGen >= 80 && m_trainer.model.step > 0) {
                m_lastGen = m_trainer.model.step;
                canvas->sample = QString::fromStdString(
                    m_trainer.model.generate(360, m_temp, m_genSeed++));
            }
            canvas->update();
            stats->setText(QString("step %1\nloss %2\n%3 params\n%4 tok/s")
                .arg(m_trainer.model.step)
                .arg(m_trainer.lastLoss, 0, 'f', 3)
                .arg(m_trainer.model.paramCount())
                .arg(int(m_trainer.tokensPerSec)));
            if (const char *df = getenv("DUMP_FRAMES")) {
                static int left = atoi(df);
                if (--left <= 0) {
                    canvas->grab().save("dump.png");
                    QApplication::quit();
                }
            }
        });
        m_timer.start(16);

        resize(1280, 820);
    }

    static int selfTest(int nSteps)
    {
        Trainer tr;
        tr.build(kCorpus);
        tr.batch = 8;
        std::printf("params: %d  vocab: %d\n",
                    tr.model.paramCount(), tr.model.vocab());
        for (int s = 0; s < nSteps; s += 10) {
            tr.trainSteps(10);
            std::printf("step %4d  loss %.4f\n",
                        tr.model.step, tr.lastLoss);
            std::fflush(stdout);
        }
        std::printf("sample: %s\n",
                    tr.model.generate(200, 0.7f, 1).c_str());
        return 0;
    }

private:
    Trainer m_trainer;
    std::string m_text = kCorpus;
    float m_temp = 0.8f;
    int m_lastGen = 0;
    uint32_t m_genSeed = 1;
    QTimer m_timer;
};

int main(int argc, char **argv)
{
    if (const char *ts = getenv("TRANS_TEST"))
        return MainWindow::selfTest(atoi(ts));

    QApplication app(argc, argv);
    app.setStyle(QStyleFactory::create("Fusion"));
    QPalette pal;
    pal.setColor(QPalette::Window, QColor(37, 37, 42));
    pal.setColor(QPalette::WindowText, QColor(220, 220, 224));
    pal.setColor(QPalette::Base, QColor(28, 28, 32));
    pal.setColor(QPalette::Text, QColor(220, 220, 224));
    pal.setColor(QPalette::Button, QColor(48, 48, 54));
    pal.setColor(QPalette::ButtonText, QColor(220, 220, 224));
    pal.setColor(QPalette::Highlight, QColor(70, 120, 200));
    app.setPalette(pal);

    MainWindow w;
    w.show();
    return app.exec();
}

#include "main.moc"
