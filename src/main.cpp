// HandmadeGPT — a tiny character-level GPT trained live, from scratch.
// One pre-LN transformer block: embeddings + learned positions, 2-head causal
// self-attention, GELU MLP, layer norms, untied unembedding. Forward, backward
// and Adam are all hand-written — no ML libraries, no autograd, just C++.
// Live visualization: loss curve, per-head attention maps, PCA of the
// character embeddings, and text sampled from the model as it learns.
// C++23 / Qt6 Widgets, no other dependencies.

#include "MainWindow.h"

#include <QApplication>
#include <QPalette>
#include <QStyleFactory>

int main(int argc, char **argv)
{
    QApplication app(argc, argv);
    app.setStyle(QStyleFactory::create(QStringLiteral("Fusion")));
    QPalette palette;
    palette.setColor(QPalette::Window, QColor(37, 37, 42));
    palette.setColor(QPalette::WindowText, QColor(220, 220, 224));
    palette.setColor(QPalette::Base, QColor(28, 28, 32));
    palette.setColor(QPalette::Text, QColor(220, 220, 224));
    palette.setColor(QPalette::Button, QColor(48, 48, 54));
    palette.setColor(QPalette::ButtonText, QColor(220, 220, 224));
    palette.setColor(QPalette::Highlight, QColor(70, 120, 200));
    app.setPalette(palette);

    MainWindow window;
    window.show();
    return app.exec();
}
