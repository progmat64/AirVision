import sys
import os
import time
import cv2
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QPushButton,
    QFileDialog,
    QComboBox,
    QVBoxLayout,
    QHBoxLayout,
    QSpinBox,
    QSlider,
    QCheckBox,
    QLineEdit,
    QTextEdit,
    QProgressBar,
    QFormLayout,
    QSizePolicy,
    QTabWidget,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QMutex, QWaitCondition
from PyQt5.QtGui import QImage, QPixmap
import argparse


class DetectionThread(QThread):
    frame_ready = pyqtSignal(QImage)
    stats_ready = pyqtSignal(int, int, float, float)
    progress = pyqtSignal(int)
    log = pyqtSignal(str, str)
    finished = pyqtSignal()

    def __init__(self, args):
        super().__init__()
        self.args = args
        self._paused = False
        self._stopped = False
        self._m = QMutex()
        self._wait = QWaitCondition()

    def pause(self):
        self._m.lock()
        self._paused = True
        self._m.unlock()

    def resume(self):
        self._m.lock()
        self._paused = False
        self._wait.wakeAll()
        self._m.unlock()

    def stop(self):
        self._m.lock()
        self._stopped = True
        if self._paused:
            self._paused = False
            self._wait.wakeAll()
        self._m.unlock()

    def run(self):
        try:
            from handler.constants import SIZE
            from handler.heatmap import HeatmapGenerator
            from handler.predict import Predictor
            from handler.track import Tracker

            cfg = vars(self.args).copy()
            cfg["imgsz"] = SIZE.get(self.args.imgsz, SIZE["s1K"])
            detector = {
                "tracking": Tracker(**cfg),
                "predict": Predictor(**cfg),
                "heatmap": HeatmapGenerator(**cfg),
            }[self.args.mode]
            detector.prepare_model()

            cap = cv2.VideoCapture(self.args.video)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
            idx = 0
            t_prev = time.time()
            fps = 0.0
            writer = None
            if self.args.save:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")

            while cap.isOpened():
                self._m.lock()
                if self._stopped:
                    self._m.unlock()
                    break
                if self._paused:
                    self._wait.wait(self._m)
                self._m.unlock()

                ok, frame = cap.read()
                if not ok:
                    break
                annotated, aux = detector.annotate_frame(frame)

                if self.args.save and writer is None:
                    h, w = annotated.shape[:2]
                    writer = cv2.VideoWriter(
                        self.args.save, fourcc, self.args.framerate, (w, h)
                    )
                if writer is not None:
                    writer.write(annotated)

                if self.args.mode == "tracking":
                    current = int(aux)
                    total_unique = len(detector.counter)
                    coverage = -1.0
                elif self.args.mode == "predict":
                    current = int(aux)
                    total_unique = -1
                    coverage = -1.0
                else:
                    current = -1
                    total_unique = -1
                    coverage = float(getattr(detector, "coverage", 0.0))

                now = time.time()
                dt = now - t_prev
                t_prev = now
                if dt > 0:
                    fps = 0.8 * fps + 0.2 * (1.0 / dt)
                self.stats_ready.emit(current, total_unique, coverage, fps)

                rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
                self.frame_ready.emit(qimg.copy())

                idx += 1
                self.progress.emit(int(idx * 100 / total))
                self.log.emit(f"Frame {idx}/{total}", "info")
                self.msleep(max(1, int(1000 / self.args.framerate)))

            cap.release()
            if writer:
                writer.release()
                self.log.emit(f"Video saved to {self.args.save}", "success")
            self.log.emit("Processing finished.", "success")
        except Exception as e:
            self.log.emit(f"Error: {e}", "error")
        finally:
            self.finished.emit()


class GUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Object Counter with YOLOv11")
        self.resize(1200, 720)
        self._build_ui()
        self._update_mode_panels(self.mode_combo.currentText())

    def _browse(self, le, filt, save=False):
        dlg = QFileDialog.getSaveFileName if save else QFileDialog.getOpenFileName
        path, _ = dlg(self, "Select file", "", filt)
        if path:
            le.setText(path)

    def _make_row(self, placeholder, filt, save=False):
        le = QLineEdit(placeholder)
        btn = QPushButton("Save…" if save else "Browse…")
        btn.clicked.connect(lambda: self._browse(le, filt, save))
        h = QHBoxLayout()
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(le)
        h.addWidget(btn)
        container = QWidget()
        container.setLayout(h)
        return le, container

    def _build_ui(self):
        # Left panel
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["tracking", "predict", "heatmap"])
        self.mode_combo.currentTextChanged.connect(self._update_mode_panels)

        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Mode:"))
        mode_layout.addWidget(self.mode_combo)
        mode_layout.addStretch(1)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._common_tab(), "Common")
        self.tabs.addTab(self._track_tab(), "Tracking")
        self.tabs.addTab(self._heat_tab(), "Heatmap")

        # Control buttons with larger size and press effects
        self.start_btn = QPushButton("Start")
        self.start_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #28a745;
                color: #fff;
                border-radius: 6px;
                padding: 12px 20px;
            }
            QPushButton:pressed {
                background-color: #218838;
            }
            """
        )
        self.pause_btn = QPushButton("Pause")
        self.pause_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #6c757d;
                color: #fff;
                border-radius: 6px;
                padding: 12px 20px;
            }
            QPushButton:pressed {
                background-color: #5a6268;
            }
            """
        )
        self.pause_btn.setEnabled(False)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #d9534f;
                color: #fff;
                border-radius: 6px;
                padding: 12px 20px;
            }
            QPushButton:pressed {
                background-color: #c9302c;
            }
            """
        )
        self.stop_btn.setEnabled(False)

        # Connect signals
        self.start_btn.clicked.connect(self._on_start)
        self.pause_btn.clicked.connect(self._on_pause)
        self.stop_btn.clicked.connect(self._on_stop)

        ctrl_layout = QHBoxLayout()
        ctrl_layout.addWidget(self.start_btn)
        ctrl_layout.addWidget(self.pause_btn)
        ctrl_layout.addWidget(self.stop_btn)

        left_layout = QVBoxLayout()
        left_layout.addLayout(mode_layout)
        left_layout.addWidget(self.tabs)
        left_layout.addLayout(ctrl_layout)
        left_layout.addStretch(1)
        left_widget = QWidget()
        left_widget.setLayout(left_layout)
        left_widget.setFixedWidth(360)

        # Right panel
        self.video_lbl = QLabel()
        self.video_lbl.setAlignment(Qt.AlignCenter)
        self.video_lbl.setStyleSheet("background:#000")
        self.video_lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.progress = QProgressBar()
        self.progress.setAlignment(Qt.AlignCenter)

        self.current = QLabel("Current: —")
        self.total = QLabel("Total: —")
        self.coverage = QLabel("Coverage: —")
        self.fps_lbl = QLabel("FPS: —")
        for lbl in (self.current, self.total, self.coverage, self.fps_lbl):
            lbl.setStyleSheet("font-weight:600; font-size:14px;")
        stats_layout = QHBoxLayout()
        stats_layout.addWidget(self.current)
        stats_layout.addWidget(self.total)
        stats_layout.addWidget(self.coverage)
        stats_layout.addWidget(self.fps_lbl)
        stats_widget = QWidget()
        stats_widget.setLayout(stats_layout)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setStyleSheet("background:#1e1e1e; color:#d1d1d1;")

        right_layout = QVBoxLayout()
        right_layout.addWidget(self.video_lbl, 5)
        right_layout.addWidget(self.progress)
        right_layout.addWidget(stats_widget)
        right_layout.addWidget(self.log, 2)
        right_widget = QWidget()
        right_widget.setLayout(right_layout)

        main_layout = QHBoxLayout(self)
        main_layout.addWidget(left_widget)
        main_layout.addWidget(right_widget)

    def _common_tab(self):
        f = QFormLayout()
        f.setLabelAlignment(Qt.AlignRight)
        self.model_le, mW = self._make_row("models/best.pt", "*.pt")
        f.addRow("Model:", mW)
        self.video_le, vW = self._make_row("data/videos/cows1.mp4", "*.mp4")
        f.addRow("Video:", vW)
        self.save_le, sW = self._make_row("data/result.mp4", "*.mp4", save=True)
        f.addRow("Output:", sW)
        self.imgsz = QComboBox()
        self.imgsz.addItems(["sd", "s1K", "s2K", "s4K"])
        f.addRow("Frame size:", self.imgsz)

        self.conf_s = QSlider(Qt.Horizontal)
        self.conf_s.setRange(1, 100)
        self.conf_s.setValue(20)
        self.conf_val = QLabel("0.20")
        self.conf_s.valueChanged.connect(
            lambda v: self.conf_val.setText(f"{v/100:.2f}")
        )
        conf_w = QWidget()
        conf_h = QHBoxLayout()
        conf_h.setContentsMargins(0, 0, 0, 0)
        conf_h.addWidget(self.conf_s)
        conf_h.addWidget(self.conf_val)
        conf_w.setLayout(conf_h)
        f.addRow("Confidence:", conf_w)

        self.skip_s = QSlider(Qt.Horizontal)
        self.skip_s.setRange(0, 100)
        self.skip_s.setValue(0)
        self.skip_val = QLabel("0")
        self.skip_s.valueChanged.connect(lambda v: self.skip_val.setText(str(v)))
        skip_w = QWidget()
        skip_h = QHBoxLayout()
        skip_h.setContentsMargins(0, 0, 0, 0)
        skip_h.addWidget(self.skip_s)
        skip_h.addWidget(self.skip_val)
        skip_w.setLayout(skip_h)
        f.addRow("Skip frames:", skip_w)

        self.start_spin = QSpinBox()
        self.start_spin.setRange(0, 10**7)
        f.addRow("Start frame:", self.start_spin)
        self.hide_cb = QCheckBox("Hide labels")
        f.addRow(self.hide_cb)
        w = QWidget()
        w.setLayout(f)
        return w

    def _track_tab(self):
        f = QFormLayout()
        f.setLabelAlignment(Qt.AlignRight)

        self.draw_cb = QCheckBox("Draw trajectories")
        self.draw_cb.stateChanged.connect(self._toggle_track)
        f.addRow(self.draw_cb)

        self.tail_s = QSlider(Qt.Horizontal)
        self.tail_s.setRange(1, 500)
        self.tail_s.setValue(50)
        self.tail_val = QLabel("50")
        self.tail_s.valueChanged.connect(lambda v: self.tail_val.setText(str(v)))
        tail_w = QWidget()
        tail_h = QHBoxLayout()
        tail_h.setContentsMargins(0, 0, 0, 0)
        tail_h.addWidget(self.tail_s)
        tail_h.addWidget(self.tail_val)
        tail_w.setLayout(tail_h)
        f.addRow("Tail length:", tail_w)

        self.thick_spin = QSpinBox()
        self.thick_spin.setRange(1, 10)
        self.thick_spin.setValue(2)
        f.addRow("Line thickness:", self.thick_spin)

        self._toggle_track()

        w = QWidget()
        w.setLayout(f)
        return w

    def _heat_tab(self):
        f = QFormLayout()
        f.setLabelAlignment(Qt.AlignRight)

        self.alpha_s = QSlider(Qt.Horizontal)
        self.alpha_s.setRange(5, 100)
        self.alpha_s.setValue(40)
        self.alpha_val = QLabel("0.40")
        self.alpha_s.valueChanged.connect(
            lambda v: self.alpha_val.setText(f"{v/100:.2f}")
        )
        alpha_w = QWidget()
        alpha_h = QHBoxLayout()
        alpha_h.setContentsMargins(0, 0, 0, 0)
        alpha_h.addWidget(self.alpha_s)
        alpha_h.addWidget(self.alpha_val)
        alpha_w.setLayout(alpha_h)
        f.addRow("Alpha:", alpha_w)

        self.radius_s = QSlider(Qt.Horizontal)
        self.radius_s.setRange(1, 100)
        self.radius_s.setValue(15)
        self.radius_val = QLabel("15")
        self.radius_s.valueChanged.connect(lambda v: self.radius_val.setText(str(v)))
        radius_w = QWidget()
        radius_h = QHBoxLayout()
        radius_h.setContentsMargins(0, 0, 0, 0)
        radius_h.addWidget(self.radius_s)
        radius_h.addWidget(self.radius_val)
        radius_w.setLayout(radius_h)
        f.addRow("Radius:", radius_w)

        self.blur_cb = QCheckBox("Blur heatmap")
        f.addRow(self.blur_cb)

        w = QWidget()
        w.setLayout(f)
        return w

    def _toggle_track(self):
        enabled = self.draw_cb.isChecked()
        self.tail_s.setEnabled(enabled)
        self.thick_spin.setEnabled(enabled)

    def _update_mode_panels(self, mode):
        self.tabs.setTabVisible(1, mode == "tracking")
        self.tabs.setTabVisible(2, mode == "heatmap")

    def _on_start(self):
        args = argparse.Namespace(
            mode=self.mode_combo.currentText(),
            model=self.model_le.text(),
            video=self.video_le.text(),
            save=self.save_le.text(),
            imgsz=self.imgsz.currentText(),
            conf=self.conf_s.value() / 100,
            skip_frames=self.skip_s.value(),
            start_frame=self.start_spin.value(),
            hide_labels=self.hide_cb.isChecked(),
            draw_lines=self.draw_cb.isChecked(),
            lines_history=self.tail_s.value(),
            line_thickness=self.thick_spin.value(),
            alpha=self.alpha_s.value() / 100,
            radius=self.radius_s.value(),
            blur=self.blur_cb.isChecked(),
            debug=False,
            init_model="GUI",
            framerate=30,
            show=False,
        )
        self.thread = DetectionThread(args)
        self.thread.frame_ready.connect(self._show_frame)
        self.thread.stats_ready.connect(self._show_stats)
        self.thread.progress.connect(self.progress.setValue)
        self.thread.log.connect(self._append_log)
        self.thread.finished.connect(self._on_finish)

        self.start_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.stop_btn.setEnabled(True)
        self.pause_btn.setText("Pause")
        self.progress.setValue(0)
        self.log.clear()
        self._reset_stats()
        self.thread.start()

    def _on_pause(self):
        if self.pause_btn.text() == "Pause":
            self.thread.pause()
            self.pause_btn.setText("Resume")
        else:
            self.thread.resume()
            self.pause_btn.setText("Pause")

    def _on_stop(self):
        self.thread.stop()

    def _on_finish(self):
        self.start_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)

    def _show_frame(self, img):
        pix = QPixmap.fromImage(img).scaled(
            self.video_lbl.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.video_lbl.setPixmap(pix)

    def _show_stats(self, curr, tot, cov, fps):
        self.current.setText(f"Current: {curr}" if curr >= 0 else "Current: —")
        self.total.setText(f"Total: {tot}" if tot >= 0 else "Total: —")
        self.coverage.setText(f"Coverage: {cov:.1f}%" if cov >= 0 else "Coverage: —")
        self.fps_lbl.setText(f"FPS: {fps:.1f}")

    def _append_log(self, msg, lvl):
        cmap = {"info": "#d1d1d1", "success": "#5cb85c", "error": "#ff5c5c"}
        color = cmap.get(lvl, "#d1d1d1")
        self.log.append(f"<span style='color:{color}'>{msg}</span>")

    def _reset_stats(self):
        self.current.setText("Current: —")
        self.total.setText("Total: —")
        self.coverage.setText("Coverage: —")
        self.fps_lbl.setText("FPS: —")


if __name__ == "__main__":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    app = QApplication(sys.argv)
    win = GUI()
    win.show()
    sys.exit(app.exec_())
