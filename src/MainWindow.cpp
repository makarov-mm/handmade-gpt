#include "MainWindow.h"

#include "Canvas.h"
#include "Corpus.h"

#include <QDoubleSpinBox>
#include <QFileDialog>
#include <QFormLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QSpinBox>

#include <fstream>
#include <iterator>

namespace {

constexpr int kSampleChars = 360;
constexpr int kSampleEverySteps = 80;
constexpr std::size_t kMaxTextBytes = 200'000;

} // namespace

MainWindow::MainWindow()
    : m_text(gpt::kCorpus)
{
    setWindowTitle(QStringLiteral("HandmadeGPT — a transformer with no libraries"));
    m_trainer.build(m_text);

    m_canvas = new Canvas(m_trainer);

    auto *lr = new QDoubleSpinBox;
    lr->setRange(1e-4, 2e-2);
    lr->setDecimals(4);
    lr->setSingleStep(5e-4);
    lr->setValue(m_trainer.lr);

    auto *batch = new QSpinBox;
    batch->setRange(2, 48);
    batch->setValue(m_trainer.batch);

    m_stepsPerFrame = new QSpinBox;
    m_stepsPerFrame->setRange(0, 20);
    m_stepsPerFrame->setValue(2);
    m_stepsPerFrame->setToolTip(QStringLiteral("0 pauses training"));

    auto *temperature = new QDoubleSpinBox;
    temperature->setRange(0.2, 2.0);
    temperature->setSingleStep(0.1);
    temperature->setValue(m_temperature);

    auto *generate = new QPushButton(QStringLiteral("Generate now"));
    auto *reset = new QPushButton(QStringLiteral("Reset model"));
    auto *load = new QPushButton(QStringLiteral("Load text file…"));

    m_stats = new QLabel;
    m_stats->setStyleSheet(QStringLiteral("color:#8fa;font-family:monospace"));

    auto *form = new QFormLayout;
    form->addRow(QStringLiteral("Learning rate"), lr);
    form->addRow(QStringLiteral("Batch"), batch);
    form->addRow(QStringLiteral("Steps/frame"), m_stepsPerFrame);
    form->addRow(QStringLiteral("Temperature"), temperature);
    form->addRow(generate);
    form->addRow(reset);
    form->addRow(load);
    form->addRow(m_stats);

    auto *group = new QGroupBox(QStringLiteral("Parameters"));
    group->setLayout(form);
    group->setFixedWidth(280);

    auto *layout = new QHBoxLayout(this);
    layout->addWidget(group);
    layout->addWidget(m_canvas, 1);

    connect(lr, &QDoubleSpinBox::valueChanged, this, [this](double v) { m_trainer.lr = static_cast<float>(v); });
    connect(batch, &QSpinBox::valueChanged, this, [this](int v) { m_trainer.batch = v; });
    connect(temperature, &QDoubleSpinBox::valueChanged, this, [this](double v) { m_temperature = static_cast<float>(v); });
    connect(generate, &QPushButton::clicked, this, &MainWindow::refreshSample);
    connect(reset, &QPushButton::clicked, this, &MainWindow::rebuild);
    connect(load, &QPushButton::clicked, this, &MainWindow::loadTextFile);
    connect(&m_timer, &QTimer::timeout, this, &MainWindow::tick);
    m_timer.start(16);

    resize(1280, 820);
}

void MainWindow::rebuild()
{
    m_trainer.build(m_text);
    m_lastSampleStep = 0;
    m_canvas->setSample(QStringLiteral("…"));
}

void MainWindow::loadTextFile()
{
    const QString fileName = QFileDialog::getOpenFileName(
        this, QStringLiteral("Training text"), {}, QStringLiteral("Text files (*.txt);;All files (*)"));
    if (fileName.isEmpty())
        return;
    std::ifstream file(fileName.toStdString(), std::ios::binary);
    std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    if (text.size() > kMaxTextBytes)
        text.resize(kMaxTextBytes);
    if (text.size() < static_cast<std::size_t>(gpt::Config::context) * 4)
        return; // too short to sample training windows from
    m_text = std::move(text);
    rebuild();
}

void MainWindow::refreshSample()
{
    m_lastSampleStep = m_trainer.model.step;
    m_canvas->setSample(QString::fromStdString(m_trainer.model.generate(kSampleChars, m_temperature, m_sampleSeed++)));
}

void MainWindow::tick()
{
    const int steps = m_stepsPerFrame->value();
    if (steps > 0)
        m_trainer.trainSteps(steps);
    if (m_trainer.model.step > 0 && m_trainer.model.step - m_lastSampleStep >= kSampleEverySteps)
        refreshSample();
    m_canvas->update();
    m_stats->setText(QStringLiteral("step %1\nloss %2\n%3 params\n%4 tok/s\n%5 threads")
                         .arg(m_trainer.model.step)
                         .arg(m_trainer.lastLoss, 0, 'f', 3)
                         .arg(m_trainer.model.paramCount())
                         .arg(static_cast<int>(m_trainer.tokensPerSec))
                         .arg(m_trainer.threadCount()));
}
