#include "Canvas.h"

#include <QImage>
#include <QPainter>
#include <QPen>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <span>
#include <vector>

namespace {

constexpr QColor kBackground(12, 12, 18);
constexpr QColor kPanel(18, 18, 26);
constexpr QColor kPanelEdge(70, 75, 90);
constexpr QColor kTitle(150, 160, 180);
constexpr QColor kLossLine(120, 230, 160);
constexpr QColor kValue(160, 170, 190);
constexpr QColor kSampleText(210, 215, 225);
constexpr QColor kSpace(240, 220, 100), kVowel(120, 220, 160), kConsonant(140, 170, 240);

// Projection of the embedding rows onto their top two principal
// components: mean-centred power iteration with one deflation step.
struct Pca2 {
    std::vector<float> x, y;
    float extent = 1e-6f; // max |coordinate|, for scaling
};

Pca2 embeddingPca(std::span<const float> E, int vocab, int d)
{
    std::vector<float> mean(static_cast<std::size_t>(d), 0.f);
    for (int v = 0; v < vocab; ++v)
        for (int i = 0; i < d; ++i) mean[static_cast<std::size_t>(i)] += E[static_cast<std::size_t>(v * d + i)];
    for (float &m : mean) m /= static_cast<float>(vocab);
    auto centred = [&](int v, int i) { return E[static_cast<std::size_t>(v * d + i)] - mean[static_cast<std::size_t>(i)]; };

    auto powerIteration = [&](std::uint32_t seed, const std::vector<float> *orthogonalTo) {
        std::mt19937 rng(seed);
        std::normal_distribution<float> gauss;
        std::vector<float> dir(static_cast<std::size_t>(d)), next(static_cast<std::size_t>(d));
        for (float &x : dir) x = gauss(rng);
        for (int it = 0; it < 12; ++it) {
            std::ranges::fill(next, 0.f);
            for (int v = 0; v < vocab; ++v) {
                float dot = 0.f;
                for (int i = 0; i < d; ++i) dot += centred(v, i) * dir[static_cast<std::size_t>(i)];
                for (int i = 0; i < d; ++i) next[static_cast<std::size_t>(i)] += dot * centred(v, i);
            }
            if (orthogonalTo) {
                float dot = 0.f;
                for (int i = 0; i < d; ++i) dot += next[static_cast<std::size_t>(i)] * (*orthogonalTo)[static_cast<std::size_t>(i)];
                for (int i = 0; i < d; ++i) next[static_cast<std::size_t>(i)] -= dot * (*orthogonalTo)[static_cast<std::size_t>(i)];
            }
            float norm = 1e-9f;
            for (float x : next) norm += x * x;
            norm = 1.f / std::sqrt(norm);
            for (int i = 0; i < d; ++i) dir[static_cast<std::size_t>(i)] = next[static_cast<std::size_t>(i)] * norm;
        }
        return dir;
    };
    const std::vector<float> p1 = powerIteration(7, nullptr);
    const std::vector<float> p2 = powerIteration(8, &p1);

    Pca2 out;
    out.x.resize(static_cast<std::size_t>(vocab));
    out.y.resize(static_cast<std::size_t>(vocab));
    for (int v = 0; v < vocab; ++v) {
        float a = 0.f, b = 0.f;
        for (int i = 0; i < d; ++i) {
            a += centred(v, i) * p1[static_cast<std::size_t>(i)];
            b += centred(v, i) * p2[static_cast<std::size_t>(i)];
        }
        out.x[static_cast<std::size_t>(v)] = a;
        out.y[static_cast<std::size_t>(v)] = b;
        out.extent = std::max({out.extent, std::fabs(a), std::fabs(b)});
    }
    return out;
}

// Label for a vocabulary character: visible stand-ins for whitespace and
// control characters, which a loaded text file can contain
QString glyphFor(char c)
{
    if (c == ' ') return QStringLiteral("_");
    if (c == '\n') return QStringLiteral("↵");
    if (c == '\t') return QStringLiteral("⇥");
    const auto u = static_cast<unsigned char>(c);
    if (u < 0x20 || u >= 0x7F) return QStringLiteral("·");
    return QString(QChar(c));
}

} // namespace

Canvas::Canvas(const gpt::Trainer &trainer, QWidget *parent)
    : QWidget(parent), m_trainer(trainer)
{
    setMinimumSize(700, 560);
}

void Canvas::setSample(QString text)
{
    m_sample = std::move(text);
    update();
}

void Canvas::paintEvent(QPaintEvent *)
{
    QPainter p(this);
    p.fillRect(rect(), kBackground);
    const int w = width(), h = height();
    const int mid = h * 55 / 100;
    drawLoss(p, QRect(8, 8, w / 2 - 12, mid - 16));
    drawAttention(p, QRect(w / 2 + 4, 8, w / 2 - 12, mid - 16));
    drawPca(p, QRect(8, mid + 4, w / 2 - 12, h - mid - 12));
    drawSample(p, QRect(w / 2 + 4, mid + 4, w / 2 - 12, h - mid - 12));
}

void Canvas::framePanel(QPainter &p, const QRect &r, const QString &title) const
{
    p.setPen(kPanelEdge);
    p.setBrush(kPanel);
    p.drawRect(r);
    p.setPen(kTitle);
    p.drawText(r.adjusted(8, 4, -8, 0), Qt::AlignTop | Qt::AlignLeft, title);
}

void Canvas::drawLoss(QPainter &p, const QRect &r) const
{
    framePanel(p, r, QStringLiteral("loss"));
    const std::vector<float> &history = m_trainer.lossHistory;
    if (history.size() < 2)
        return;
    const auto [lo, hi] = std::ranges::minmax(history);
    const float floor = std::min(lo, 0.5f);
    const float range = std::max(0.1f, hi - floor);
    const QRect in = r.adjusted(8, 22, -8, -8);
    p.setPen(QPen(kLossLine, 1.5));
    QPointF prev;
    for (std::size_t i = 0; i < history.size(); ++i) {
        const float fx = static_cast<float>(in.x()) + static_cast<float>(in.width()) * static_cast<float>(i) / static_cast<float>(history.size() - 1);
        const float fy = static_cast<float>(in.bottom()) - static_cast<float>(in.height()) * (history[i] - floor) / range;
        const QPointF point(fx, fy);
        if (i > 0)
            p.drawLine(prev, point);
        prev = point;
    }
    p.setPen(kValue);
    p.drawText(in, Qt::AlignTop | Qt::AlignRight, QString::number(m_trainer.lastLoss, 'f', 3));
}

void Canvas::drawAttention(QPainter &p, const QRect &r) const
{
    framePanel(p, r, QStringLiteral("attention  head 0 / head 1"));
    const std::vector<float> &A = m_trainer.lastAttn;
    constexpr int T = gpt::Config::context, H = gpt::Config::heads;
    if (A.size() < static_cast<std::size_t>(H) * T * T)
        return;
    const QRect in = r.adjusted(8, 22, -8, -8);
    const int cell = std::max(1, std::min(in.width() / (H * T + 4), in.height() / T));
    for (int h = 0; h < H; ++h) {
        const int ox = in.x() + h * (cell * T + 8);
        QImage img(T, T, QImage::Format_RGB32);
        for (int t = 0; t < T; ++t) {
            auto *row = reinterpret_cast<QRgb *>(img.scanLine(t));
            for (int s = 0; s < T; ++s) {
                const float a = std::clamp(A[static_cast<std::size_t>(h * T * T + t * T + s)] * 3.f, 0.f, 1.f);
                row[s] = h == 0 ? qRgb(static_cast<int>(30 + 200 * a), static_cast<int>(30 + 120 * a), 40)
                                : qRgb(40, static_cast<int>(30 + 140 * a), static_cast<int>(30 + 210 * a));
            }
        }
        p.drawImage(QRect(ox, in.y(), cell * T, cell * T), img);
    }
}

void Canvas::drawPca(QPainter &p, const QRect &r) const
{
    framePanel(p, r, QStringLiteral("embedding PCA"));
    const gpt::Model &m = m_trainer.model;
    const int V = m.vocab(), d = gpt::Config::width;
    if (V == 0)
        return;
    const std::span<const float> E(m.par.w.data() + m.par.emb, static_cast<std::size_t>(V * d));
    const Pca2 pca = embeddingPca(E, V, d);

    const QRect in = r.adjusted(10, 24, -10, -10);
    QFont f = p.font();
    f.setPointSize(9);
    f.setBold(true);
    p.setFont(f);
    for (int v = 0; v < V; ++v) {
        const float fx = static_cast<float>(in.center().x()) + pca.x[static_cast<std::size_t>(v)] / pca.extent * static_cast<float>(in.width()) * 0.46f;
        const float fy = static_cast<float>(in.center().y()) - pca.y[static_cast<std::size_t>(v)] / pca.extent * static_cast<float>(in.height()) * 0.46f;
        const char c = m.itoc(v);
        const bool vowel = std::strchr("aeiou", c) != nullptr && c != '\0';
        p.setPen(c == ' ' ? kSpace : vowel ? kVowel : kConsonant);
        p.drawText(QPointF(fx, fy), glyphFor(c));
    }
}

void Canvas::drawSample(QPainter &p, const QRect &r) const
{
    framePanel(p, r, QStringLiteral("sampled text"));
    p.setPen(kSampleText);
    QFont f(QStringLiteral("monospace"));
    f.setPointSize(9);
    p.setFont(f);
    p.drawText(r.adjusted(10, 26, -10, -8), Qt::TextWordWrap | Qt::AlignTop, m_sample);
}
