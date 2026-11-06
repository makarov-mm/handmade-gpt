// The four live panels: loss curve, attention maps, embedding PCA, sample.
#pragma once

#include "Trainer.h"

#include <QString>
#include <QWidget>

class Canvas : public QWidget {
    Q_OBJECT
public:
    explicit Canvas(const gpt::Trainer &trainer, QWidget *parent = nullptr);

    void setSample(QString text);

protected:
    void paintEvent(QPaintEvent *) override;

private:
    const gpt::Trainer &m_trainer;
    QString m_sample = QStringLiteral("…");

    void framePanel(QPainter &p, const QRect &r, const QString &title) const;
    void drawLoss(QPainter &p, const QRect &r) const;
    void drawAttention(QPainter &p, const QRect &r) const;
    void drawPca(QPainter &p, const QRect &r) const;
    void drawSample(QPainter &p, const QRect &r) const;
};
