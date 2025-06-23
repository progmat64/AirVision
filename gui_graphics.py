# gui.py
import sys
import os
import time
import cv2
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QFileDialog,
    QComboBox, QVBoxLayout, QHBoxLayout, QSpinBox, QSlider,
    QCheckBox, QLineEdit, QTextEdit, QProgressBar,
    QFormLayout, QSizePolicy, QTabWidget
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QMutex, QWaitCondition
from PyQt5.QtGui import QImage, QPixmap
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import argparse

class DetectionThread(QThread):
    frame_ready  = pyqtSignal(QImage)
    stats_ready  = pyqtSignal(int, int, float, float)
    progress     = pyqtSignal(int)
    log          = pyqtSignal(str, str)
    finished     = pyqtSignal()

    def __init__(self, args):
        super().__init__()
        self.args    = args
        self._paused = False
        self._stopped= False
        self._m      = QMutex()
        self._wait   = QWaitCondition()

    def pause(self):
        self._m.lock(); self._paused = True; self._m.unlock()

    def resume(self):
        self._m.lock(); self._paused = False; self._wait.wakeAll(); self._m.unlock()

    def stop(self):
        self._m.lock(); self._stopped = True
        if self._paused:
            self._paused = False; self._wait.wakeAll()
        self._m.unlock()

    def run(self):
        try:
            from handler.constants import SIZE
            from handler.heatmap   import HeatmapGenerator
            from handler.predict   import Predictor
            from handler.track     import Tracker

            cfg = vars(self.args).copy()
            cfg["imgsz"] = SIZE.get(self.args.imgsz, SIZE["s1K"])
            detector = {
                "tracking": Tracker(**cfg),
                "predict" : Predictor(**cfg),
                "heatmap" : HeatmapGenerator(**cfg),
            }[self.args.mode]
            detector.prepare_model()

            cap    = cv2.VideoCapture(self.args.video)
            total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
            idx    = 0
            t_prev = time.time(); fps = 0.0
            writer = None
            if self.args.save:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")

            while cap.isOpened():
                self._m.lock()
                if self._stopped: self._m.unlock(); break
                if self._paused:  self._wait.wait(self._m)
                self._m.unlock()

                ok, frame = cap.read()
                if not ok: break

                annotated, aux = detector.annotate_frame(frame)

                if self.args.save and writer is None:
                    h, w = annotated.shape[:2]
                    writer = cv2.VideoWriter(self.args.save, fourcc, self.args.framerate, (w, h))
                if writer:
                    writer.write(annotated)

                # unpack stats
                if   self.args.mode == "tracking":
                    curr, total_u, cov = int(aux), len(detector.counter), -1.0
                elif self.args.mode == "predict":
                    curr, total_u, cov = int(aux), -1, -1.0
                else:
                    curr, total_u, cov = -1, -1, float(getattr(detector, "coverage", 0.0))

                now = time.time(); dt = now - t_prev; t_prev = now
                if dt>0: fps = 0.8*fps + 0.2*(1.0/dt)
                self.stats_ready.emit(curr, total_u, cov, fps)

                # send image
                rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                h,w,ch = rgb.shape
                qimg = QImage(rgb.data, w, h, ch*w, QImage.Format_RGB888)
                self.frame_ready.emit(qimg.copy())

                idx += 1
                self.progress.emit(int(idx*100/total))
                self.log.emit(f"Frame {idx}/{total}", "info")
                self.msleep(max(1, int(1000/self.args.framerate)))

            cap.release()
            if writer:
                writer.release()
                self.log.emit(f"Video saved to {self.args.save}", "success")
            self.log.emit("Processing finished.", "success")

        except Exception as e:
            self.log.emit(f"Error: {e}", "error")
        finally:
            self.finished.emit()

class GraphCanvas(FigureCanvas):
    def __init__(self, title: str, ylabel: str):
        fig = Figure(figsize=(4,2))
        super().__init__(fig)
        self.ax    = fig.add_subplot(111)
        self.ax.set_title(title)
        self.ax.set_ylabel(ylabel)
        self.xdata = []
        self.ydata = []
        self.line, = self.ax.plot([], [], 'b-')

    def add_data(self, x: int, y: float):
        self.xdata.append(x)
        self.ydata.append(y)
        self.line.set_data(self.xdata, self.ydata)
        self.ax.relim()
        self.ax.autoscale_view()
        self.draw()

class GUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Object Counter with YOLOv8")
        self.resize(1400, 800)
        self.frame_idx = 0
        self._build_ui()
        self._update_mode_panels(self.mode_combo.currentText())

    def _build_ui(self):
        # --- Left Panel ---
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["tracking","predict","heatmap"])
        self.mode_combo.currentTextChanged.connect(self._update_mode_panels)

        form = QFormLayout()
        self.model_le, model_w = self._make_row("models/best.pt","*.pt")
        form.addRow("Model:", model_w)
        self.video_le, video_w = self._make_row("data/videos/cows1.mp4","*.mp4")
        form.addRow("Video:", video_w)
        self.save_le,  save_w  = self._make_row("data/result.mp4","*.mp4", save=True)
        form.addRow("Output:", save_w)

        self.imgsz = QComboBox(); self.imgsz.addItems(["sd","s1K","s2K","s4K"])
        form.addRow("Frame size:", self.imgsz)

        # confidence slider
        self.conf_s = QSlider(Qt.Horizontal); self.conf_s.setRange(1,100); self.conf_s.setValue(20)
        self.conf_val = QLabel("0.20"); self.conf_s.valueChanged.connect(lambda v: self.conf_val.setText(f"{v/100:.2f}"))
        cbox = QHBoxLayout(); cbox.addWidget(self.conf_s); cbox.addWidget(self.conf_val)
        cfw = QWidget(); cfw.setLayout(cbox); form.addRow("Confidence:", cfw)

        # skip slider
        self.skip_s = QSlider(Qt.Horizontal); self.skip_s.setRange(0,100); self.skip_s.setValue(0)
        self.skip_val = QLabel("0"); self.skip_s.valueChanged.connect(lambda v: self.skip_val.setText(str(v)))
        sbox = QHBoxLayout(); sbox.addWidget(self.skip_s); sbox.addWidget(self.skip_val)
        sfw = QWidget(); sfw.setLayout(sbox); form.addRow("Skip frames:", sfw)

        self.start_spin = QSpinBox(); self.start_spin.setRange(0,10**7)
        form.addRow("Start frame:", self.start_spin)

        self.hide_cb = QCheckBox("Hide labels")
        form.addRow(self.hide_cb)

        self.tabs = QTabWidget()
        w_common = QWidget(); w_common.setLayout(form)
        self.tabs.addTab(w_common, "Common")

        # Tracking tab
        track_f = QFormLayout()
        self.draw_cb = QCheckBox("Draw trajectories"); track_f.addRow(self.draw_cb)
        self.tail_s  = QSlider(Qt.Horizontal); self.tail_s.setRange(1,500); self.tail_s.setValue(50)
        self.tail_val= QLabel("50"); self.tail_s.valueChanged.connect(lambda v: self.tail_val.setText(str(v)))
        tbox = QHBoxLayout(); tbox.addWidget(self.tail_s); tbox.addWidget(self.tail_val)
        tfw = QWidget(); tfw.setLayout(tbox); track_f.addRow("Tail length:", tfw)
        self.thick_spin = QSpinBox(); self.thick_spin.setRange(1,10); self.thick_spin.setValue(2)
        track_f.addRow("Line thickness:", self.thick_spin)
        w_track = QWidget(); w_track.setLayout(track_f)
        self.tabs.addTab(w_track, "Tracking")

        # Heatmap tab
        heat_f = QFormLayout()
        self.alpha_s = QSlider(Qt.Horizontal); self.alpha_s.setRange(5,100); self.alpha_s.setValue(40)
        self.alpha_val= QLabel("0.40"); self.alpha_s.valueChanged.connect(lambda v: self.alpha_val.setText(f"{v/100:.2f}"))
        abox = QHBoxLayout(); abox.addWidget(self.alpha_s); abox.addWidget(self.alpha_val)
        afw = QWidget(); afw.setLayout(abox); heat_f.addRow("Alpha:", afw)

        self.radius_s = QSlider(Qt.Horizontal); self.radius_s.setRange(1,100); self.radius_s.setValue(15)
        self.radius_val= QLabel("15"); self.radius_s.valueChanged.connect(lambda v: self.radius_val.setText(str(v)))
        rbox = QHBoxLayout(); rbox.addWidget(self.radius_s); rbox.addWidget(self.radius_val)
        rfw = QWidget(); rfw.setLayout(rbox); heat_f.addRow("Radius:", rfw)

        self.blur_cb = QCheckBox("Blur heatmap"); heat_f.addRow(self.blur_cb)
        w_heat = QWidget(); w_heat.setLayout(heat_f)
        self.tabs.addTab(w_heat, "Heatmap")

        # control buttons
        self.start_btn = QPushButton("Start")
        self.pause_btn = QPushButton("Pause"); self.pause_btn.setEnabled(False)
        self.stop_btn  = QPushButton("Stop");  self.stop_btn.setEnabled(False)
        btns = QHBoxLayout(); btns.addWidget(self.start_btn); btns.addWidget(self.pause_btn); btns.addWidget(self.stop_btn)

        left = QVBoxLayout()
        left.addWidget(QLabel("Mode:")); left.addWidget(self.mode_combo)
        left.addWidget(self.tabs); left.addLayout(btns); left.addStretch(1)
        left_widget = QWidget(); left_widget.setLayout(left); left_widget.setFixedWidth(360)

        # --- Right Panel ---
        self.video_lbl = QLabel()
        self.video_lbl.setAlignment(Qt.AlignCenter)
        self.video_lbl.setStyleSheet("background:#000")
        self.video_lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.progress = QProgressBar(); self.progress.setAlignment(Qt.AlignCenter)

        self.current = QLabel("Current: —")
        self.total   = QLabel("Total: —")
        self.coverage= QLabel("Coverage: —")
        self.fps_lbl = QLabel("FPS: —")
        for lbl in (self.current,self.total,self.coverage,self.fps_lbl):
            lbl.setStyleSheet("font-weight:600; font-size:14px;")

        stats = QHBoxLayout()
        stats.addWidget(self.current); stats.addWidget(self.total)
        stats.addWidget(self.coverage); stats.addWidget(self.fps_lbl)
        stats_widget = QWidget(); stats_widget.setLayout(stats)

        self.log = QTextEdit(); self.log.setReadOnly(True)
        self.log.setStyleSheet("background:#1e1e1e; color:#d1d1d1")

        # Graphs
        self.graph_current = GraphCanvas("Current Count", "#objects")
        self.graph_total   = GraphCanvas("Total Unique", "count")
        self.graph_cov     = GraphCanvas("Coverage", "%")
        self.graph_fps     = GraphCanvas("FPS", "fps")
        glay = QHBoxLayout()
        for g in (self.graph_current,self.graph_total,self.graph_cov,self.graph_fps):
            glay.addWidget(g)

        right = QVBoxLayout()
        right.addWidget(self.video_lbl, 5)
        right.addWidget(self.progress)
        right.addWidget(stats_widget)
        right.addLayout(glay)
        right.addWidget(self.log, 2)
        right_widget = QWidget(); right_widget.setLayout(right)

        # main
        main = QHBoxLayout(self)
        main.addWidget(left_widget)
        main.addWidget(right_widget)

        # connections
        self.start_btn.clicked.connect(self._on_start)
        self.pause_btn.clicked.connect(self._on_pause)
        self.stop_btn.clicked.connect(self._on_stop)

    def _make_row(self, placeholder, filt, save=False):
        le = QLineEdit(placeholder)
        btn= QPushButton("Save…" if save else "Browse…")
        btn.clicked.connect(lambda: self._browse(le, filt, save))
        hl = QHBoxLayout(); hl.addWidget(le); hl.addWidget(btn)
        w  = QWidget(); w.setLayout(hl)
        return le, w

    def _update_mode_panels(self, mode):
        self.tabs.setTabVisible(1, mode=="tracking")
        self.tabs.setTabVisible(2, mode=="heatmap")

    def _on_start(self):
        args = argparse.Namespace(
            mode=self.mode_combo.currentText(), model=self.model_le.text(),
            video=self.video_le.text(), save=self.save_le.text(),
            imgsz=self.imgsz.currentText(), conf=self.conf_s.value()/100,
            skip_frames=self.skip_s.value(), start_frame=self.start_spin.value(),
            hide_labels=self.hide_cb.isChecked(), draw_lines=self.draw_cb.isChecked(),
            lines_history=self.tail_s.value(), line_thickness=self.thick_spin.value(),
            alpha=self.alpha_s.value()/100, radius=self.radius_s.value(),
            blur=self.blur_cb.isChecked(), debug=False, init_model="GUI",
            framerate=30, show=False
        )
        self.frame_idx = 0
        for g in (self.graph_current,self.graph_total,self.graph_cov,self.graph_fps):
            g.xdata.clear(); g.ydata.clear(); g.line.set_data([],[]); g.ax.relim(); g.ax.autoscale_view()

        self.thread = DetectionThread(args)
        self.thread.frame_ready.connect(self._show_frame)
        self.thread.stats_ready.connect(self._show_stats)
        self.thread.progress.connect(self.progress.setValue)
        self.thread.log.connect(self._append_log)
        self.thread.finished.connect(self._on_finish)
        self.thread.start()

        self.start_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.stop_btn.setEnabled(True)

    def _on_pause(self):
        if self.thread._paused:
            self.thread.resume()
            self.pause_btn.setText("Pause")
        else:
            self.thread.pause()
            self.pause_btn.setText("Resume")

    def _on_stop(self):
        self.thread.stop()

    def _on_finish(self):
        self.start_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)

    def _show_frame(self, img):
        pix = QPixmap.fromImage(img).scaled(self.video_lbl.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_lbl.setPixmap(pix)

    def _show_stats(self, curr, tot, cov, fps):
        self.frame_idx += 1
        self.current.setText(f"Current: {curr}" if curr>=0 else "Current: —")
        self.total.setText(f"Total: {tot}" if tot>=0 else "Total: —")
        self.coverage.setText(f"Coverage: {cov:.1f}%" if cov>=0 else "Coverage: —")
        self.fps_lbl.setText(f"FPS: {fps:.1f}")
        self.graph_current.add_data(self.frame_idx, curr if curr>=0 else 0)
        self.graph_total.add_data(self.frame_idx, tot if tot>=0 else 0)
        self.graph_cov.add_data(self.frame_idx, cov if cov>=0 else 0)
        self.graph_fps.add_data(self.frame_idx, fps)

    def _append_log(self, msg, lvl):
        cmap = {"info":"#d1d1d","success":"#5cb85c","error":"#ff5c5c"}
        self.log.append(f"<span style='color:{cmap.get(lvl,'#d1d1d')}'>{msg}</span>")

    def _browse(self, le, filt, save=False):
        dlg = QFileDialog.getSaveFileName if save else QFileDialog.getOpenFileName
        path,_ = dlg(self,"Select file","",filt)
        if path: le.setText(path)

if __name__ == "__main__":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    app = QApplication(sys.argv)
    win = GUI()
    win.show()
    sys.exit(app.exec_())
