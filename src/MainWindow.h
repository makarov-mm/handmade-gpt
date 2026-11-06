// Parameter panel, training timer and the canvas.
#pragma once

#include "Trainer.h"

#include <QTimer>
#include <QWidget>

#include <cstdint>
#include <string>

class Canvas;
class QLabel;
class QSpinBox;

class MainWindow : public QWidget {
    Q_OBJECT
public:
    MainWindow();

private:
    gpt::Trainer m_trainer;
    std::string m_text;          // current training text
    float m_temperature = 0.8f;
    int m_lastSampleStep = 0;    // step at which the sample panel was last refreshed
    std::uint32_t m_sampleSeed = 1;
    QTimer m_timer;
    Canvas *m_canvas = nullptr;
    QSpinBox *m_stepsPerFrame = nullptr;
    QLabel *m_stats = nullptr;

    void rebuild();              // retrains from scratch on m_text
    void loadTextFile();
    void refreshSample();
    void tick();
};
