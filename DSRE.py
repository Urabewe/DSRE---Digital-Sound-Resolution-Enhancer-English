import os
import sys
import traceback
from typing import Optional

import subprocess
import soundfile as sf
import tempfile

import numpy as np
from scipy import signal
import librosa
import resampy

from PySide6 import QtCore, QtWidgets
from PySide6.QtGui import QIcon, QTextCursor

def add_ffmpeg_to_path():
    if hasattr(sys, "_MEIPASS"):  # Temporary directory after packaging
        ffmpeg_dir = os.path.join(sys._MEIPASS, "ffmpeg")
    else:
        ffmpeg_dir = os.path.join(os.path.dirname(__file__), "ffmpeg")
    os.environ["PATH"] += os.pathsep + ffmpeg_dir

add_ffmpeg_to_path()

def save_wav24_out(in_path, y_out, sr, out_path, fmt="ALAC", normalize=True):
    import tempfile, subprocess, numpy as np, soundfile as sf, os

    # Ensure shape is (n, ch)
    if y_out.ndim == 1:
        data = y_out[:, None]
    else:
        data = y_out.T if y_out.shape[0] < y_out.shape[1] else y_out

    # Convert to float32 and normalize
    data = data.astype(np.float32, copy=False)
    if normalize:
        peak = float(np.max(np.abs(data)))
        if peak > 1.0:
            data /= peak
    else:
        data = np.clip(data, -1.0, 1.0)

    # Temporary WAV file
    tmp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp_wav.close()
    sf.write(tmp_wav.name, data, sr, subtype="FLOAT")

    fmt = fmt.upper()
    out_path = os.path.splitext(out_path)[0] + (".m4a" if fmt == "ALAC" else ".flac")

    codec_map = {"ALAC": "alac", "FLAC": "flac"}
    sample_fmt_map = {"ALAC": "s32p", "FLAC": "s32"}  # Force 24bit integer

    if fmt == "ALAC":
        cmd = [
            "ffmpeg", "-y",
            "-i", tmp_wav.name,
            "-i", in_path,
            "-map", "0:a",       # Temporary WAV audio
            "-map", "1:v?",      # Cover
            "-map_metadata", "1",# Metadata
            "-c:a", codec_map[fmt],
            "-sample_fmt", sample_fmt_map[fmt],
            "-c:v", "copy",
            out_path
        ]
    elif fmt == "FLAC":
        # Extract cover image
        cover_tmp = None
        try:
            cover_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            cover_tmp.close()
            subprocess.run(
                ["ffmpeg", "-y", "-i", in_path, "-an", "-c:v", "copy", cover_tmp.name],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        except Exception:
            cover_tmp = None

        if cover_tmp and os.path.exists(cover_tmp.name):
            cmd = [
                "ffmpeg", "-y",
                "-i", tmp_wav.name,  # WAV audio
                "-i", in_path,       # Metadata source
                "-i", cover_tmp.name, # Cover
                "-map", "0:a",       # Audio
                "-map", "2:v",       # Cover
                "-disposition:v", "attached_pic",
                "-map_metadata", "1",# Metadata
                "-c:a", codec_map[fmt],
                "-sample_fmt", sample_fmt_map[fmt],
                "-c:v", "copy",
                out_path
            ]
        else:
            cmd = [
                "ffmpeg", "-y",
                "-i", tmp_wav.name,
                "-i", in_path,
                "-map", "0:a",
                "-map_metadata", "1",
                "-c:a", codec_map[fmt],
                "-sample_fmt", sample_fmt_map[fmt],
                out_path
            ]

    subprocess.run(cmd, check=True)
    os.remove(tmp_wav.name)
    if fmt == "FLAC" and cover_tmp and os.path.exists(cover_tmp.name):
        os.remove(cover_tmp.name)

    return out_path

# ======== DSP: SSB Single Sideband Frequency Shift ========
def freq_shift_mono(x: np.ndarray, f_shift: float, d_sr: float) -> np.ndarray:
    N_orig = len(x)
    # Pad to power of 2 for efficient FFT/Hilbert implementation
    N_padded = 1 << int(np.ceil(np.log2(max(1, N_orig))))
    S_hilbert = signal.hilbert(np.hstack((x, np.zeros(N_padded - N_orig, dtype=x.dtype))))
    S_factor = np.exp(2j * np.pi * f_shift * d_sr * np.arange(0, N_padded))
    return (S_hilbert * S_factor)[:N_orig].real

def freq_shift_multi(x: np.ndarray, f_shift: float, d_sr: float) -> np.ndarray:
    return np.asarray([freq_shift_mono(x[i], f_shift, d_sr) for i in range(len(x))])

def zansei_impl(
    x: np.ndarray,
    sr: int,
    m: int = 8,
    decay: float = 1.25,
    pre_hp: float = 3000.0,
    post_hp: float = 16000.0,
    filter_order: int = 11,
    progress_cb=None,
    abort_cb=None,  # New callback
) -> np.ndarray:
    # Pre-processing high-pass
    b, a = signal.butter(filter_order, pre_hp / (sr / 2), 'highpass')
    d_src = signal.filtfilt(b, a, x)

    d_sr = 1.0 / sr
    f_dn = freq_shift_mono if (x.ndim == 1) else freq_shift_multi
    d_res = np.zeros_like(x)

    for i in range(m):
        if abort_cb and abort_cb():
            break  # Exit processing immediately
        shift_hz = sr * (i + 1) / (m * 2.0)
        d_res += f_dn(d_src, shift_hz, d_sr) * np.exp(-(i + 1) * decay)
        if progress_cb:
            progress_cb(i + 1, m)

    # Post-processing high-pass
    b, a = signal.butter(filter_order, post_hp / (sr / 2), 'highpass')
    d_res = signal.filtfilt(b, a, d_res)

    adp_power = float(np.mean(np.abs(d_res)))
    src_power = float(np.mean(np.abs(x)))
    adj_factor = src_power / (adp_power + src_power + 1e-12)

    y = (x + d_res) * adj_factor
    return y

# ======== Background Worker Thread ========
class DSREWorker(QtCore.QThread):
    sig_log = QtCore.Signal(str)                         # Text log
    sig_file_progress = QtCore.Signal(int, int, str)     # Current file progress (cur, total, filename)
    sig_step_progress = QtCore.Signal(int, str)          # Single file internal progress(0~100), filename
    sig_overall_progress = QtCore.Signal(int, int)       # Overall progress (done, total)
    sig_file_done = QtCore.Signal(str, str)              # Single file completed (in_path, out_path)
    sig_error = QtCore.Signal(str, str)                  # Error (filename, err_msg)
    sig_finished = QtCore.Signal()                       # All completed

    def __init__(self, files, output_dir, params, parent=None):
        super().__init__(parent)
        self.files = files
        self.output_dir = output_dir
        self.params = params
        self._abort = False

    def abort(self):
        self._abort = True

    def run(self):
        total = len(self.files)
        done = 0
        self.sig_overall_progress.emit(done, total)

        for idx, in_path in enumerate(self.files, start=1):
            if self._abort:
                break

            fname = os.path.basename(in_path)
            self.sig_file_progress.emit(idx, total, fname)
            self.sig_step_progress.emit(0, fname)

            try:
                # Read
                self.sig_log.emit(f"Loading: {in_path}")
                y, sr = librosa.load(in_path, mono=False, sr=None)

                # Align to (ch, n)
                if y.ndim == 1:
                    y = y[np.newaxis, :]
                # Resample
                target_sr = int(self.params["target_sr"])
                if sr != target_sr:
                    self.sig_log.emit(f"Processing: {fname}: {sr} -> {target_sr}")
                    y = resampy.resample(y, sr, target_sr, filter='kaiser_fast')
                    sr = target_sr

                # Process
                def step_cb(cur, m):
                    pct = int(cur * 100 / max(1, m))
                    self.sig_step_progress.emit(pct, fname)

                y_out = zansei_impl(
                    y, sr,
                    m=int(self.params["m"]),
                    decay=float(self.params["decay"]),
                    pre_hp=float(self.params["pre_hp"]),
                    post_hp=float(self.params["post_hp"]),
                    filter_order=int(self.params["filter_order"]),
                    progress_cb=step_cb,
                    abort_cb=lambda: self._abort  # Pass abort callback
                )

                # Save (preserve original format + metadata)
                os.makedirs(self.output_dir, exist_ok=True)
                base, ext = os.path.splitext(fname)

                out_path = os.path.join(self.output_dir,
                                        f"{base}.{self.params['format'].lower() if self.params['format'] == 'flac' else 'm4a'}")
                out_path = save_wav24_out(in_path, y_out, sr, out_path, fmt=self.params['format'])

                self.sig_log.emit(f"File saved: {out_path}")
                self.sig_file_done.emit(in_path, out_path)

            except Exception as e:
                err = "".join(traceback.format_exception_only(type(e), e)).strip()
                self.sig_error.emit(fname, err)
                self.sig_log.emit(f"[Error] {fname}: {err}")

            done += 1
            self.sig_overall_progress.emit(done, total)
            self.sig_step_progress.emit(100, fname)

        self.sig_finished.emit()

# ======== GUI ========
class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DSRE v1.1.250908_beta")

        # 获取相对路径的图标
        icon_path = os.path.join(os.path.dirname(__file__), "logo.ico")
        self.setWindowIcon(QIcon(icon_path))

        self.resize(900, 600)

        # File list
        self.list_files = QtWidgets.QListWidget()
        self.btn_add = QtWidgets.QPushButton("Add Input Files")
        self.btn_clear = QtWidgets.QPushButton("Clear Input List")
        self.btn_outdir = QtWidgets.QPushButton("Select Output Directory")
        self.le_outdir = QtWidgets.QLineEdit()
        self.le_outdir.setPlaceholderText("Output folder")
        self.le_outdir.setText(os.path.abspath("output"))

        # Parameters
        self.sb_m = QtWidgets.QSpinBox()
        self.sb_m.setRange(1, 1024)
        self.sb_m.setValue(8)
        self.dsb_decay = QtWidgets.QDoubleSpinBox()
        self.dsb_decay.setRange(0.0, 1024)
        self.dsb_decay.setSingleStep(0.05)
        self.dsb_decay.setValue(1.25)
        self.sb_pre = QtWidgets.QSpinBox()
        self.sb_pre.setRange(1, 384000)
        self.sb_pre.setValue(3000)
        self.sb_post = QtWidgets.QSpinBox()
        self.sb_post.setRange(1, 384000)
        self.sb_post.setValue(16000)
        self.sb_order = QtWidgets.QSpinBox()
        self.sb_order.setRange(1, 1000)
        self.sb_order.setValue(11)
        self.sb_sr = QtWidgets.QSpinBox()
        self.sb_sr.setRange(1, 384000)
        self.sb_sr.setSingleStep(1000)
        self.sb_sr.setValue(96000)

        # Progress
        self.pb_file = QtWidgets.QProgressBar()    # Single file progress
        self.pb_all = QtWidgets.QProgressBar()     # Overall progress
        self.lbl_now = QtWidgets.QLabel("Control")

        # Control buttons
        self.btn_start = QtWidgets.QPushButton("Start Processing")
        self.btn_cancel = QtWidgets.QPushButton("Cancel Processing")
        self.btn_cancel.setEnabled(False)

        # Log
        self.te_log = QtWidgets.QTextEdit()
        self.te_log.setReadOnly(True)

        # ===== Layout =====
        grid = QtWidgets.QGridLayout()

        # === Left column: Input files ===
        vleft = QtWidgets.QVBoxLayout()
        lbl_files = QtWidgets.QLabel("Input Files")
        lbl_files.setAlignment(QtCore.Qt.AlignHCenter)
        vleft.addWidget(lbl_files)
        vleft.addWidget(self.list_files)
        grid.addLayout(vleft, 0, 0, 7, 1)

        # === Middle column: Operations ===
        vmid = QtWidgets.QVBoxLayout()
        lbl_ops = QtWidgets.QLabel("Operations")
        lbl_ops.setAlignment(QtCore.Qt.AlignHCenter)
        vmid.addWidget(lbl_ops)

        vbtn = QtWidgets.QVBoxLayout()
        vbtn.addWidget(self.btn_add)
        vbtn.addWidget(self.btn_clear)
        vbtn.addSpacing(10)
        vbtn.addWidget(QtWidgets.QLabel("Output Directory"))
        vbtn.addWidget(self.le_outdir)
        vbtn.addWidget(self.btn_outdir)
        vbtn.addSpacing(20)

        # Place lbl_now ("Control") here
        vbtn.addWidget(self.lbl_now)

        vbtn.addWidget(self.btn_start)
        vbtn.addWidget(self.btn_cancel)
        vbtn.addStretch(1)

        # Output format selection
        self.cb_format = QtWidgets.QComboBox()
        self.cb_format.addItems(["ALAC", "FLAC"])  # Two optional formats
        vbtn.addWidget(QtWidgets.QLabel("Output Encoding Format"))
        vbtn.addWidget(self.cb_format)

        vmid.addLayout(vbtn)
        grid.addLayout(vmid, 0, 1, 7, 1)

        # === Right column: Parameter settings + Progress ===
        vright = QtWidgets.QVBoxLayout()
        lbl_params = QtWidgets.QLabel("Parameter Settings")
        lbl_params.setAlignment(QtCore.Qt.AlignHCenter)
        vright.addWidget(lbl_params)

        form = QtWidgets.QFormLayout()
        form.addRow("Modulation Count:", self.sb_m)
        form.addRow("Decay Amplitude:", self.dsb_decay)
        form.addRow("Pre-processing High-pass Filter Cutoff Frequency (Hz):", self.sb_pre)
        form.addRow("Post-processing High-pass Filter Cutoff Frequency (Hz):", self.sb_post)
        form.addRow("Filter Order:", self.sb_order)
        form.addRow("Target Sample Rate (Hz):", self.sb_sr)
        vright.addLayout(form)

        vright.addSpacing(20)

        vprog = QtWidgets.QVBoxLayout()
        vprog.addWidget(QtWidgets.QLabel("Current File Processing Progress"))
        vprog.addWidget(self.pb_file)
        vprog.addWidget(QtWidgets.QLabel("Overall File Processing Progress"))
        vprog.addWidget(self.pb_all)
        vprog.addStretch(1)
        vright.addLayout(vprog)
        grid.addLayout(vright, 0, 2, 7, 1)

        # === Bottom log ===
        grid.addWidget(QtWidgets.QLabel("Log"), 7, 0)
        grid.addWidget(self.te_log, 8, 0, 1, 3)

        self.setLayout(grid)

        # Connect signals
        self.btn_add.clicked.connect(self.on_add_files)
        self.btn_clear.clicked.connect(self.list_files.clear)
        self.btn_outdir.clicked.connect(self.on_choose_outdir)
        self.btn_start.clicked.connect(self.on_start)
        self.btn_cancel.clicked.connect(self.on_cancel)

        self.worker: Optional[DSREWorker] = None

        # Write welcome message after initialization
        self.append_log("Software by: Qu Le Fan")
        self.append_log("Feedback: Le_Fan_Qv@outlook.com")
        self.append_log("Discussion Group: 323861356 (QQ)")

    def on_add_files(self):
        filters = (
            "Audio Files (*.wav *.mp3 *.m4a *.flac *.ogg *.aiff *.aif *.aac *.wma *.mka);;"
            "All Files (*.*)"
        )
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "Select Input Files", "", filters)
        for f in files:
            if f and (self.list_files.findItems(f, QtCore.Qt.MatchFlag.MatchExactly) == []):
                self.list_files.addItem(f)

    def on_choose_outdir(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Output Directory", self.le_outdir.text() or "")
        if d:
            self.le_outdir.setText(d)

    def params(self):
        return dict(
            m=self.sb_m.value(),
            decay=self.dsb_decay.value(),
            pre_hp=self.sb_pre.value(),
            post_hp=self.sb_post.value(),
            target_sr=self.sb_sr.value(),
            filter_order=self.sb_order.value(),
            bit_depth=24,  # Fixed output 24bit
            format=self.cb_format.currentText()  # ALAC or FLAC
        )

    def append_log(self, s: str):
        self.te_log.append(s)
        self.te_log.moveCursor(QTextCursor.End)

    def on_start(self):
        files = [self.list_files.item(i).text() for i in range(self.list_files.count())]
        if not files:
            QtWidgets.QMessageBox.warning(self, "No Files", "Please add at least one input file")
            return
        outdir = self.le_outdir.text().strip() or os.path.abspath("output")

        # Reset progress
        self.pb_all.setValue(0)
        self.pb_file.setValue(0)
        self.lbl_now.setText("Initializing...")
        self.append_log(f"Starting to process {len(files)} files...")

        # Lock buttons
        self.btn_start.setEnabled(False)
        self.btn_cancel.setEnabled(True)

        # Start background thread
        self.worker = DSREWorker(files, outdir, self.params())
        self.worker.sig_log.connect(self.append_log)
        self.worker.sig_file_progress.connect(self.on_file_progress)
        self.worker.sig_step_progress.connect(self.on_step_progress)
        self.worker.sig_overall_progress.connect(self.on_overall_progress)
        self.worker.sig_file_done.connect(self.on_file_done)
        self.worker.sig_error.connect(self.on_error)
        self.worker.sig_finished.connect(self.on_finished)
        self.worker.start()

    @QtCore.Slot(int, int, str)
    def on_file_progress(self, cur, total, fname):
        self.lbl_now.setText(f"Processing... [{cur}/{total}]: {fname}")
        self.pb_file.setValue(0)

    @QtCore.Slot(int, str)
    def on_step_progress(self, pct, fname):
        self.pb_file.setValue(pct)

    @QtCore.Slot(int, int)
    def on_overall_progress(self, done, total):
        pct = int(done * 100 / max(1, total))
        self.pb_all.setValue(pct)

    @QtCore.Slot(str, str)
    def on_file_done(self, in_path, out_path):
        self.append_log(f"Processing completed: {os.path.basename(in_path)} -> {out_path}")

    @QtCore.Slot(str, str)
    def on_error(self, fname, err):
        self.append_log(f"[Error] {fname}: {err}")

    def on_cancel(self):
        if self.worker and self.worker.isRunning():
            self.append_log("Cancelling...")
            self.worker.abort()

    def on_finished(self):
        self.append_log("All files have been processed")
        self.lbl_now.setText("Control")
        self.btn_start.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.worker = None

def main():

    import ctypes
    myappid = "com.lefanqv.dsre"  # Your custom application ID, must be a string
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

    app = QtWidgets.QApplication(sys.argv)

    # Globally set application icon
    icon_path = os.path.join(os.path.dirname(__file__), "logo.ico")
    app.setWindowIcon(QIcon(icon_path))

    w = MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
