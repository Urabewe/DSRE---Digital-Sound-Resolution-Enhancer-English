import os
import sys
import traceback
import json
import time
from typing import Optional, Dict, Any, List, Tuple

import subprocess
import soundfile as sf
import tempfile

import numpy as np
from scipy import signal
import librosa
import resampy

from PySide6 import QtCore, QtWidgets
from PySide6.QtGui import QIcon, QTextCursor, QDragEnterEvent, QDropEvent, QKeySequence, QAction

def add_ffmpeg_to_path():
    if hasattr(sys, "_MEIPASS"):  # Temporary directory after packaging
        ffmpeg_dir = os.path.join(sys._MEIPASS, "ffmpeg")
    else:
        ffmpeg_dir = os.path.join(os.path.dirname(__file__), "ffmpeg")
    
    # Check if ffmpeg directory exists
    if not os.path.exists(ffmpeg_dir):
        print(f"Warning: FFmpeg directory not found: {ffmpeg_dir}")
        print("Please ensure FFmpeg is installed in the 'ffmpeg' directory next to this script.")
    else:
        ffmpeg_exe = os.path.join(ffmpeg_dir, "ffmpeg.exe")
        if not os.path.exists(ffmpeg_exe):
            print(f"Warning: ffmpeg.exe not found in: {ffmpeg_dir}")
        else:
            print(f"FFmpeg found: {ffmpeg_exe}")
    
    os.environ["PATH"] += os.pathsep + ffmpeg_dir

add_ffmpeg_to_path()

def save_wav24_out(in_path, y_out, sr, out_path, fmt="ALAC", normalize=True):
    import tempfile, subprocess, numpy as np, soundfile as sf, os

    # Input validation
    if not os.path.exists(in_path):
        raise FileNotFoundError(f"Input file not found: {in_path}")
    if y_out is None or y_out.size == 0:
        raise ValueError("Empty audio data provided")
    if sr <= 0:
        raise ValueError(f"Invalid sample rate: {sr}")
    if fmt.upper() not in ["ALAC", "FLAC", "MP3"]:
        raise ValueError(f"Unsupported format: {fmt}")

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

    # Temporary WAV file with proper cleanup
    tmp_wav = None
    cover_tmp = None
    try:
        tmp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp_wav.close()
        sf.write(tmp_wav.name, data, sr, subtype="FLOAT")

        fmt = fmt.upper()
        # Set correct file extension based on format
        if fmt == "ALAC":
            out_path = os.path.splitext(out_path)[0] + ".m4a"
        elif fmt == "FLAC":
            out_path = os.path.splitext(out_path)[0] + ".flac"
        elif fmt == "MP3":
            out_path = os.path.splitext(out_path)[0] + ".mp3"

        codec_map = {"ALAC": "alac", "FLAC": "flac", "MP3": "libmp3lame"}
        sample_fmt_map = {"ALAC": "s32p", "FLAC": "s32", "MP3": "s16p"}  # MP3 uses 16bit planar

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
        elif fmt == "MP3":
            # MP3 encoding with high quality settings
            cmd = [
                "ffmpeg", "-y",
                "-i", tmp_wav.name,
                "-i", in_path,
                "-map", "0:a",       # Temporary WAV audio
                "-map", "1:v?",      # Cover (optional)
                "-map_metadata", "1",# Metadata
                "-c:a", codec_map[fmt],
                "-sample_fmt", sample_fmt_map[fmt],
                "-b:a", "320k",      # High quality bitrate
            "-c:v", "copy",
            out_path
        ]
        elif fmt == "FLAC":
                # Extract cover image
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

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            raise Exception(f"FFmpeg command failed: {' '.join(cmd)}. Error: {e}")
        except FileNotFoundError as e:
            raise Exception(f"FFmpeg not found. Please ensure FFmpeg is installed in the 'ffmpeg' directory next to this script. Error: {e}")
        
        return out_path

    finally:
        # Cleanup temporary files
        if tmp_wav and os.path.exists(tmp_wav.name):
            try:
                os.remove(tmp_wav.name)
            except OSError:
                pass
        if cover_tmp and os.path.exists(cover_tmp.name):
            try:
                os.remove(cover_tmp.name)
            except OSError:
                pass

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
    progress_cb: Optional[callable] = None,
    abort_cb: Optional[callable] = None,
) -> np.ndarray:
    """
    Implement the Zansei audio enhancement algorithm.
    
    Args:
        x: Input audio data (channels, samples)
        sr: Sample rate in Hz
        m: Number of modulation iterations
        decay: Decay factor for high frequencies
        pre_hp: Pre-processing high-pass filter cutoff (Hz)
        post_hp: Post-processing high-pass filter cutoff (Hz)
        filter_order: Butterworth filter order
        progress_cb: Optional callback for progress updates (current, total)
        abort_cb: Optional callback to check for abort signal
        
    Returns:
        Enhanced audio data with same shape as input
    """
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

# ======== Custom List Widget with Drag & Drop ========
class DragDropListWidget(QtWidgets.QListWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.DropOnly)
        self.setDefaultDropAction(QtCore.Qt.DropAction.CopyAction)
        
        # Add placeholder text
        self.placeholder_item = QtWidgets.QListWidgetItem("Drag and drop audio files here...")
        self.placeholder_item.setFlags(QtCore.Qt.ItemFlag.NoItemFlags)  # Make it non-selectable
        self.addItem(self.placeholder_item)
        
        # Enable drag and drop explicitly
        self.setAcceptDrops(True)
        
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            audio_extensions = {'.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aiff', '.aif', '.aac', '.wma', '.mka'}
            for url in urls:
                file_path = url.toLocalFile()
                if os.path.isfile(file_path):
                    _, ext = os.path.splitext(file_path.lower())
                    if ext in audio_extensions:
                        event.acceptProposedAction()
                        return
        event.ignore()
    
    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()
    
    def dropEvent(self, event: QDropEvent):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            audio_extensions = {'.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aiff', '.aif', '.aac', '.wma', '.mka'}
            
            # Remove placeholder if it exists
            if self.count() == 1 and self.item(0) == self.placeholder_item:
                self.takeItem(0)
            
            for url in urls:
                file_path = url.toLocalFile()
                if os.path.isfile(file_path):
                    _, ext = os.path.splitext(file_path.lower())
                    if ext in audio_extensions:
                        # Add file if not already in list
                        if not self.findItems(file_path, QtCore.Qt.MatchFlag.MatchExactly):
                            self.addItem(file_path)
            
            event.acceptProposedAction()
        else:
            event.ignore()

# ======== Background Worker Thread ========
class DSREWorker(QtCore.QThread):
    sig_log = QtCore.Signal(str)                         # Text log
    sig_file_progress = QtCore.Signal(int, int, str)     # Current file progress (cur, total, filename)
    sig_step_progress = QtCore.Signal(int, str)          # Single file internal progress(0~100), filename
    sig_overall_progress = QtCore.Signal(int, int)       # Overall progress (done, total)
    sig_file_done = QtCore.Signal(str, str)              # Single file completed (in_path, out_path)
    sig_error = QtCore.Signal(str, str)                  # Error (filename, err_msg)
    sig_finished = QtCore.Signal()                       # All completed
    sig_retry_available = QtCore.Signal(str, str)        # Retry available (filename, error)
    sig_processing_stats = QtCore.Signal(dict)           # Processing statistics

    def __init__(self, files, output_dir, params, parent=None):
        super().__init__(parent)
        self.files = files
        self.output_dir = output_dir
        self.params = params
        self._abort = False
        self.processing_stats = {
            'total_files': len(files),
            'processed_files': 0,
            'failed_files': 0,
            'total_size_mb': 0,
            'processed_size_mb': 0,
            'start_time': None,
            'estimated_remaining': 0
        }

    def abort(self):
        self._abort = True
    
    def get_file_size_mb(self, file_path: str) -> float:
        """Get file size in MB"""
        try:
            return os.path.getsize(file_path) / (1024 * 1024)
        except OSError:
            return 0.0
    
    def estimate_processing_time(self, file_size_mb: float) -> float:
        """Estimate processing time based on file size (seconds)"""
        # Rough estimation: ~1MB per second for processing
        return max(1.0, file_size_mb * 0.5)
    
    def process_audio_chunked(self, y: np.ndarray, sr: int, chunk_size: int = 44100 * 10) -> np.ndarray:
        """Process audio in chunks for large files"""
        if len(y) <= chunk_size:
            return zansei_impl(
                y, sr,
                m=int(self.params["m"]),
                decay=float(self.params["decay"]),
                pre_hp=float(self.params["pre_hp"]),
                post_hp=float(self.params["post_hp"]),
                filter_order=int(self.params["filter_order"]),
                progress_cb=None,
                abort_cb=lambda: self._abort
            )
        
        # Process in chunks
        chunks = []
        total_chunks = (len(y) + chunk_size - 1) // chunk_size
        
        for i in range(0, len(y), chunk_size):
            if self._abort:
                break
                
            chunk = y[i:i + chunk_size]
            if len(chunk) > 0:
                processed_chunk = zansei_impl(
                    chunk, sr,
                    m=int(self.params["m"]),
                    decay=float(self.params["decay"]),
                    pre_hp=float(self.params["pre_hp"]),
                    post_hp=float(self.params["post_hp"]),
                    filter_order=int(self.params["filter_order"]),
                    progress_cb=None,
                    abort_cb=lambda: self._abort
                )
                chunks.append(processed_chunk)
        
        return np.concatenate(chunks) if chunks else y
    
    def load_audio_with_recovery(self, file_path: str) -> Tuple[np.ndarray, int]:
        """Load audio with multiple fallback strategies for corrupted files"""
        try:
            # First attempt: Normal loading
            y, sr = librosa.load(file_path, mono=False, sr=None)
            return y, sr
        except Exception as e1:
            self.sig_log.emit(f"Primary load failed, trying recovery methods...")
            
            try:
                # Second attempt: Force mono and resample
                self.sig_log.emit(f"Attempting mono recovery...")
                y, sr = librosa.load(file_path, mono=True, sr=None)
                return y, sr
            except Exception as e2:
                try:
                    # Third attempt: Use soundfile directly with error handling
                    self.sig_log.emit(f"Attempting direct soundfile loading...")
                    import soundfile as sf
                    y, sr = sf.read(file_path, always_2d=True)
                    if y.ndim == 1:
                        y = y[:, np.newaxis]
                    return y.T, sr  # Convert to (channels, samples)
                except Exception as e3:
                    try:
                        # Fourth attempt: Load with different parameters
                        self.sig_log.emit(f"Attempting alternative loading parameters...")
                        y, sr = librosa.load(file_path, mono=False, sr=44100)  # Force 44.1kHz
                        return y, sr
                    except Exception as e4:
                        # Final attempt: Create silence as fallback
                        self.sig_log.emit(f"All recovery methods failed, creating silent audio...")
                        # Create 1 second of silence at 44.1kHz
                        sr = 44100
                        y = np.zeros((1, sr), dtype=np.float32)
                        return y, sr
    
    def categorize_error(self, error: Exception) -> str:
        """Categorize errors for better retry handling"""
        error_str = str(error).lower()
        
        # Fatal errors - don't retry
        if any(keyword in error_str for keyword in ['permission denied', 'access denied', 'disk full', 'no space']):
            return "fatal"
        
        # I/O errors - retry with longer delay
        if any(keyword in error_str for keyword in ['file not found', 'no such file', 'network', 'timeout', 'connection']):
            return "io"
        
        # Memory errors - retry with chunked processing
        if any(keyword in error_str for keyword in ['memory', 'out of memory', 'allocation']):
            return "memory"
        
        # Audio format errors - retry with different parameters
        if any(keyword in error_str for keyword in ['format', 'codec', 'sample rate', 'bitrate']):
            return "format"
        
        # FFmpeg errors - retry
        if any(keyword in error_str for keyword in ['ffmpeg', 'encoder', 'decoder']):
            return "ffmpeg"
        
        # Default - retry
        return "retry"

    def run(self):
        total = len(self.files)
        done = 0
        self.processing_stats['start_time'] = time.time()
        
        # Calculate total file sizes for better progress estimation
        total_size = sum(self.get_file_size_mb(f) for f in self.files)
        self.processing_stats['total_size_mb'] = total_size
        
        self.sig_overall_progress.emit(done, total)
        self.sig_processing_stats.emit(self.processing_stats.copy())

        for idx, in_path in enumerate(self.files, start=1):
            if self._abort:
                break

            fname = os.path.basename(in_path)
            file_size_mb = self.get_file_size_mb(in_path)
            
            self.sig_file_progress.emit(idx, total, fname)
            self.sig_step_progress.emit(0, fname)

            # Estimate processing time
            estimated_time = self.estimate_processing_time(file_size_mb)
            self.sig_log.emit(f"Processing {fname} ({file_size_mb:.1f}MB, est. {estimated_time:.1f}s)")

            retry_count = 0
            max_retries = 3
            
            while retry_count <= max_retries:
                try:
                    # Read with error recovery
                    self.sig_log.emit(f"Loading: {in_path}")
                    y, sr = self.load_audio_with_recovery(in_path)

                    # Align to (ch, n)
                    if y.ndim == 1:
                        y = y[np.newaxis, :]
                    
                    # Resample
                    target_sr = int(self.params["target_sr"])
                    if sr != target_sr:
                        self.sig_log.emit(f"Processing: {fname}: {sr} -> {target_sr}")
                        y = resampy.resample(y, sr, target_sr, filter='kaiser_fast')
                        sr = target_sr

                    # Process with chunked processing for large files
                    def step_cb(cur, m):
                        pct = int(cur * 100 / max(1, m))
                        self.sig_step_progress.emit(pct, fname)

                    # Use chunked processing for files > 50MB
                    if file_size_mb > 50:
                        self.sig_log.emit(f"Using chunked processing for large file: {fname}")
                        y_out = self.process_audio_chunked(y, sr)
                    else:
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

                    # Set correct file extension based on format
                    format_ext = {
                        'ALAC': 'm4a',
                        'FLAC': 'flac', 
                        'MP3': 'mp3'
                    }
                    ext = format_ext.get(self.params['format'], 'm4a')
                    out_path = os.path.join(self.output_dir, f"{base}.{ext}")
                    out_path = save_wav24_out(in_path, y_out, sr, out_path, fmt=self.params['format'])

                    self.sig_log.emit(f"File saved: {out_path}")
                    self.sig_file_done.emit(in_path, out_path)
                    
                    # Update statistics
                    self.processing_stats['processed_files'] += 1
                    self.processing_stats['processed_size_mb'] += file_size_mb
                    
                    break  # Success, exit retry loop

                except Exception as e:
                    err = "".join(traceback.format_exception_only(type(e), e)).strip()
                    retry_count += 1
                    
                    # Categorize error for better handling
                    error_type = self.categorize_error(e)
                    
                    if retry_count <= max_retries and error_type != "fatal":
                        self.sig_log.emit(f"[Retry {retry_count}/{max_retries}] {fname}: {err}")
                        # Longer delay for certain error types
                        delay = 2 if error_type == "io" else 1
                        time.sleep(delay)
                    else:
                        self.sig_error.emit(fname, err)
                        self.sig_log.emit(f"[Error] {fname}: {err}")
                        self.processing_stats['failed_files'] += 1
                        if error_type != "fatal":
                            self.sig_retry_available.emit(fname, err)

            done += 1
            self.sig_overall_progress.emit(done, total)
            self.sig_step_progress.emit(100, fname)
            
            # Update processing statistics
            self.sig_processing_stats.emit(self.processing_stats.copy())

        self.sig_finished.emit()

# ======== GUI ========
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DSRE v1.1.250908_beta")

        # Get relative path icon
        icon_path = os.path.join(os.path.dirname(__file__), "logo.ico")
        self.setWindowIcon(QIcon(icon_path))

        self.resize(1000, 700)
        
        # UI state
        self.dark_mode = False
        self.recent_files = []
        self.max_recent_files = 10
        
        # Create central widget and main layout
        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        
        # Create menu bar
        self.create_menu_bar()
        
        # Create status bar
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Ready")

        # File list with drag & drop support
        self.list_files = DragDropListWidget()
        self.list_files.setToolTip("Drag and drop audio files here, or use the 'Add Input Files' button")
        
        # Ensure the widget can receive focus and events
        self.list_files.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.list_files.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        
        self.btn_add = QtWidgets.QPushButton("Add Input Files")
        self.btn_clear = QtWidgets.QPushButton("Clear Input List")
        self.btn_remove_selected = QtWidgets.QPushButton("Remove Selected")
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
        
        # Processing statistics
        self.lbl_stats = QtWidgets.QLabel("Ready to process")
        self.lbl_stats.setStyleSheet("QLabel { color: #666; font-size: 10px; }")
        self.lbl_eta = QtWidgets.QLabel("")
        self.lbl_eta.setStyleSheet("QLabel { color: #666; font-size: 10px; }")

        # Control buttons
        self.btn_start = QtWidgets.QPushButton("Start Processing")
        self.btn_cancel = QtWidgets.QPushButton("Cancel Processing")
        self.btn_cancel.setEnabled(False)
        self.btn_retry = QtWidgets.QPushButton("Retry Failed Files")
        self.btn_retry.setEnabled(False)
        
        # Darkmode button
        self.btn_test_dark = QtWidgets.QPushButton("Dark Mode")
        self.btn_test_dark.clicked.connect(self.toggle_dark_mode)
        self.btn_retry.setStyleSheet("QPushButton { background-color: #ff9800; color: white; }")

        # Log
        self.te_log = QtWidgets.QTextEdit()
        self.te_log.setReadOnly(True)

        # ===== Layout with Resizable Panels =====
        # Create main splitter (horizontal)
        main_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        
        # === Left panel: Input files ===
        left_widget = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout()
        lbl_files = QtWidgets.QLabel("Input Files")
        lbl_files.setAlignment(QtCore.Qt.AlignHCenter)
        left_layout.addWidget(lbl_files)
        left_layout.addWidget(self.list_files)
        left_widget.setLayout(left_layout)
        main_splitter.addWidget(left_widget)
        
        # === Middle panel: Operations ===
        middle_widget = QtWidgets.QWidget()
        middle_layout = QtWidgets.QVBoxLayout()
        lbl_ops = QtWidgets.QLabel("Operations")
        lbl_ops.setAlignment(QtCore.Qt.AlignHCenter)
        middle_layout.addWidget(lbl_ops)

        vbtn = QtWidgets.QVBoxLayout()
        vbtn.addWidget(self.btn_add)
        vbtn.addWidget(self.btn_clear)
        vbtn.addWidget(self.btn_remove_selected)
        vbtn.addSpacing(10)
        vbtn.addWidget(QtWidgets.QLabel("Output Directory"))
        vbtn.addWidget(self.le_outdir)
        vbtn.addWidget(self.btn_outdir)
        vbtn.addSpacing(20)

        # Place lbl_now ("Control") here
        vbtn.addWidget(self.lbl_now)

        vbtn.addWidget(self.btn_start)
        vbtn.addWidget(self.btn_cancel)
        vbtn.addWidget(self.btn_retry)
        vbtn.addWidget(self.btn_test_dark)
        vbtn.addStretch(1)

        # Output format selection
        self.cb_format = QtWidgets.QComboBox()
        self.cb_format.addItems(["ALAC", "FLAC", "MP3"])  # Three optional formats
        vbtn.addWidget(QtWidgets.QLabel("Output Encoding Format"))
        vbtn.addWidget(self.cb_format)

        middle_layout.addLayout(vbtn)
        middle_widget.setLayout(middle_layout)
        main_splitter.addWidget(middle_widget)

        # === Right panel: Parameter settings + Progress ===
        right_widget = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout()
        lbl_params = QtWidgets.QLabel("Parameter Settings")
        lbl_params.setAlignment(QtCore.Qt.AlignHCenter)
        right_layout.addWidget(lbl_params)

        form = QtWidgets.QFormLayout()
        form.addRow("Modulation Count:", self.sb_m)
        form.addRow("Decay Amplitude:", self.dsb_decay)
        form.addRow("Pre-processing High-pass Filter Cutoff Frequency (Hz):", self.sb_pre)
        form.addRow("Post-processing High-pass Filter Cutoff Frequency (Hz):", self.sb_post)
        form.addRow("Filter Order:", self.sb_order)
        form.addRow("Target Sample Rate (Hz):", self.sb_sr)
        right_layout.addLayout(form)

        right_layout.addSpacing(20)

        vprog = QtWidgets.QVBoxLayout()
        vprog.addWidget(QtWidgets.QLabel("Current File Processing Progress"))
        vprog.addWidget(self.pb_file)
        vprog.addWidget(QtWidgets.QLabel("Overall File Processing Progress"))
        vprog.addWidget(self.pb_all)
        vprog.addWidget(self.lbl_stats)
        vprog.addWidget(self.lbl_eta)
        vprog.addStretch(1)
        right_layout.addLayout(vprog)
        right_widget.setLayout(right_layout)
        main_splitter.addWidget(right_widget)
        
        # Set initial sizes for panels (proportional)
        main_splitter.setSizes([300, 250, 350])
        
        # Create vertical splitter for main content and log
        vertical_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        vertical_splitter.addWidget(main_splitter)
        
        # === Bottom panel: Log ===
        log_widget = QtWidgets.QWidget()
        log_layout = QtWidgets.QVBoxLayout()
        log_layout.addWidget(QtWidgets.QLabel("Log"))
        log_layout.addWidget(self.te_log)
        log_widget.setLayout(log_layout)
        vertical_splitter.addWidget(log_widget)
        
        # Set initial sizes for vertical splitter
        vertical_splitter.setSizes([500, 200])
        
        # Set the main layout
        main_layout = QtWidgets.QVBoxLayout()
        main_layout.addWidget(vertical_splitter)
        self.central_widget.setLayout(main_layout)

        # Connect signals
        self.btn_add.clicked.connect(self.on_add_files)
        self.btn_clear.clicked.connect(self.on_clear_files)
        self.btn_remove_selected.clicked.connect(self.on_remove_selected)
        self.btn_outdir.clicked.connect(self.on_choose_outdir)
        self.btn_start.clicked.connect(self.on_start)
        self.btn_cancel.clicked.connect(self.on_cancel)
        self.btn_retry.clicked.connect(self.on_retry_failed)
        
        # Dark mode toggle is connected in button creation

        self.worker: Optional[DSREWorker] = None
        self.config_file = os.path.join(os.path.dirname(__file__), "dsre_config.json")
        self.failed_files = []  # Track failed files for retry
        
        # Load saved configuration
        self.load_config()
        
        # Connect parameter changes to auto-save
        self.sb_m.valueChanged.connect(self.save_config)
        self.dsb_decay.valueChanged.connect(self.save_config)
        self.sb_pre.valueChanged.connect(self.save_config)
        self.sb_post.valueChanged.connect(self.save_config)
        self.sb_order.valueChanged.connect(self.save_config)
        self.sb_sr.valueChanged.connect(self.save_config)
        self.le_outdir.textChanged.connect(self.save_config)
        self.cb_format.currentTextChanged.connect(self.save_config)

        # Write welcome message after initialization
        self.append_log("Software by: Qu Le Fan")
        self.append_log("Feedback: Le_Fan_Qv@outlook.com")
        self.append_log("Discussion Group: 323861356 (QQ)")
        
        # Apply initial theme after all widgets are created
        self.apply_theme()
        
        # Set initial button text
        if self.dark_mode:
            self.btn_test_dark.setText("Light Mode")
        else:
            self.btn_test_dark.setText("Dark Mode")
    
    def create_menu_bar(self):
        """Create menu bar with keyboard shortcuts"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('&File')
        
        # Add files action
        add_files_action = QAction('&Add Files...', self)
        add_files_action.setShortcut(QKeySequence.StandardKey.Open)
        add_files_action.triggered.connect(self.on_add_files)
        file_menu.addAction(add_files_action)
        
        # Clear files action
        clear_files_action = QAction('&Clear All', self)
        clear_files_action.setShortcut('Ctrl+L')
        clear_files_action.triggered.connect(self.on_clear_files)
        file_menu.addAction(clear_files_action)
        
        file_menu.addSeparator()
        
        # Recent files submenu
        self.recent_menu = file_menu.addMenu('&Recent Files')
        self.update_recent_files_menu()
        
        file_menu.addSeparator()
        
        # Exit action
        exit_action = QAction('E&xit', self)
        exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        
        # Processing menu
        process_menu = menubar.addMenu('&Processing')
        
        # Start processing action
        start_action = QAction('&Start Processing', self)
        start_action.setShortcut('F5')
        start_action.triggered.connect(self.on_start)
        process_menu.addAction(start_action)
        
        # Cancel processing action
        cancel_action = QAction('&Cancel Processing', self)
        cancel_action.setShortcut('Escape')
        cancel_action.triggered.connect(self.on_cancel)
        process_menu.addAction(cancel_action)
        
        # Retry failed files action
        retry_action = QAction('&Retry Failed Files', self)
        retry_action.setShortcut('Ctrl+R')
        retry_action.triggered.connect(self.on_retry_failed)
        process_menu.addAction(retry_action)
        
        # Help menu
        help_menu = menubar.addMenu('&Help')
        
        # About action
        about_action = QAction('&About', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def toggle_dark_mode(self):
        """Toggle between light and dark mode"""
        self.dark_mode = not self.dark_mode
        self.apply_theme()
        self.save_config()  # Save the dark mode setting
        
        # Update button text to reflect current state
        if self.dark_mode:
            self.btn_test_dark.setText("Light Mode")
        else:
            self.btn_test_dark.setText("Dark Mode")
    
    def apply_theme(self):
        """Apply light or dark theme"""
        if self.dark_mode:
            # Dark theme
            self.setStyleSheet("""
                QMainWindow {
                    background-color: #2b2b2b;
                    color: #ffffff;
                }
                QWidget {
                    background-color: #2b2b2b;
                    color: #ffffff;
                }
                QListWidget {
                    background-color: #3c3c3c;
                    color: #ffffff;
                    border: 2px dashed #666666;
                    border-radius: 5px;
                }
                QListWidget::item {
                    background-color: transparent;
                    padding: 5px;
                    border-bottom: 1px solid #555555;
                }
                QListWidget::item:hover {
                    background-color: #4a4a4a;
                }
                QListWidget::item:selected {
                    background-color: #0078d4;
                    color: white;
                }
                QPushButton {
                    background-color: #404040;
                    color: #ffffff;
                    border: 1px solid #666666;
                    padding: 5px;
                    border-radius: 3px;
                }
                QPushButton:hover {
                    background-color: #505050;
                }
                QPushButton:pressed {
                    background-color: #606060;
                }
                QLineEdit {
                    background-color: #3c3c3c;
                    color: #ffffff;
                    border: 1px solid #666666;
                    padding: 5px;
                }
                QComboBox {
                    background-color: #3c3c3c;
                    color: #ffffff;
                    border: 1px solid #666666;
                    padding: 5px;
                }
                QSpinBox, QDoubleSpinBox {
                    background-color: #3c3c3c;
                    color: #ffffff;
                    border: 1px solid #666666;
                    padding: 5px;
                }
                QProgressBar {
                    background-color: #3c3c3c;
                    border: 1px solid #666666;
                    text-align: center;
                }
                QProgressBar::chunk {
                    background-color: #0078d4;
                }
                QTextEdit {
                    background-color: #3c3c3c;
                    color: #ffffff;
                    border: 1px solid #666666;
                }
                QLabel {
                    color: #ffffff;
                }
                QMenuBar {
                    background-color: #2b2b2b;
                    color: #ffffff;
                    border-bottom: 1px solid #666666;
                }
                QMenuBar::item {
                    background-color: transparent;
                    padding: 4px 8px;
                }
                QMenuBar::item:selected {
                    background-color: #404040;
                }
                QMenu {
                    background-color: #3c3c3c;
                    color: #ffffff;
                    border: 1px solid #666666;
                }
                QMenu::item {
                    padding: 4px 20px;
                }
                QMenu::item:selected {
                    background-color: #404040;
                }
                QStatusBar {
                    background-color: #2b2b2b;
                    color: #ffffff;
                    border-top: 1px solid #666666;
                }
                QSplitter::handle {
                    background-color: #666666;
                }
            """)
        else:
            # Light theme (default)
            self.setStyleSheet("""
                QMainWindow {
                    background-color: #ffffff;
                    color: #333333;
                }
                QWidget {
                    background-color: #ffffff;
                    color: #333333;
                }
                QListWidget {
                    border: 2px dashed #aaa;
                    border-radius: 5px;
                    background-color: #f9f9f9;
                    min-height: 200px;
                }
                QListWidget::item {
                    padding: 5px;
                    border-bottom: 1px solid #eee;
                    color: #333;
                    background-color: transparent;
                }
                QListWidget::item:hover {
                    background-color: #e3f2fd;
                    color: #333;
                }
                QListWidget::item:selected {
                    background-color: #2196f3;
                    color: white;
                    border: 1px solid #1976d2;
                }
                QListWidget::item:selected:hover {
                    background-color: #1976d2;
                    color: white;
                }
                QPushButton {
                    background-color: #f0f0f0;
                    color: #333333;
                    border: 1px solid #cccccc;
                    padding: 5px;
                    border-radius: 3px;
                }
                QPushButton:hover {
                    background-color: #e0e0e0;
                }
                QPushButton:pressed {
                    background-color: #d0d0d0;
                }
                QLineEdit {
                    background-color: #ffffff;
                    color: #333333;
                    border: 1px solid #cccccc;
                    padding: 5px;
                }
                QComboBox {
                    background-color: #ffffff;
                    color: #333333;
                    border: 1px solid #cccccc;
                    padding: 5px;
                }
                QSpinBox, QDoubleSpinBox {
                    background-color: #ffffff;
                    color: #333333;
                    border: 1px solid #cccccc;
                    padding: 5px;
                }
                QProgressBar {
                    background-color: #f0f0f0;
                    border: 1px solid #cccccc;
                    text-align: center;
                }
                QProgressBar::chunk {
                    background-color: #2196f3;
                }
                QTextEdit {
                    background-color: #ffffff;
                    color: #333333;
                    border: 1px solid #cccccc;
                }
                QLabel {
                    color: #333333;
                }
                QMenuBar {
                    background-color: #f0f0f0;
                    color: #333333;
                    border-bottom: 1px solid #cccccc;
                }
                QMenuBar::item {
                    background-color: transparent;
                    padding: 4px 8px;
                }
                QMenuBar::item:selected {
                    background-color: #e0e0e0;
                }
                QMenu {
                    background-color: #ffffff;
                    color: #333333;
                    border: 1px solid #cccccc;
                }
                QMenu::item {
                    padding: 4px 20px;
                }
                QMenu::item:selected {
                    background-color: #e0e0e0;
                }
                QStatusBar {
                    background-color: #f0f0f0;
                    color: #333333;
                    border-top: 1px solid #cccccc;
                }
                QSplitter::handle {
                    background-color: #cccccc;
                }
            """)
    
    def show_about(self):
        """Show about dialog"""
        QtWidgets.QMessageBox.about(self, "About DSRE", 
            "DSRE - Digital Sound Resolution Enhancer\n\n"
            "Version: 1.1.250908_beta\n"
            "Software by: Qu Le Fan\n"
            "Feedback: Le_Fan_Qv@outlook.com\n"
            "Discussion Group: 323861356 (QQ)\n\n"
            "A high-performance audio enhancement tool that can batch-convert "
            "any audio files into high-resolution audio using non-deep-learning "
            "frequency enhancement algorithms.")

    def on_add_files(self):
        filters = (
            "Audio Files (*.wav *.mp3 *.m4a *.flac *.ogg *.aiff *.aif *.aac *.wma *.mka);;"
            "All Files (*.*)"
        )
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "Select Input Files", "", filters)
        
        # Remove placeholder if it exists
        if self.list_files.count() == 1 and self.list_files.item(0).flags() == QtCore.Qt.ItemFlag.NoItemFlags:
            self.list_files.takeItem(0)
        
        for f in files:
            if f and (self.list_files.findItems(f, QtCore.Qt.MatchFlag.MatchExactly) == []):
                self.list_files.addItem(f)
                # Add to recent files
                self.add_to_recent_files(f)

    def on_choose_outdir(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Output Directory", self.le_outdir.text() or "")
        if d:
            self.le_outdir.setText(d)
    
    def on_clear_files(self):
        """Clear all files from the list"""
        self.list_files.clear()
        # Add placeholder back
        placeholder_item = QtWidgets.QListWidgetItem("Drag and drop audio files here...")
        placeholder_item.setFlags(QtCore.Qt.ItemFlag.NoItemFlags)
        self.list_files.addItem(placeholder_item)
    
    def on_remove_selected(self):
        """Remove selected files from the list"""
        current_row = self.list_files.currentRow()
        if current_row >= 0:
            self.list_files.takeItem(current_row)
            # Add placeholder back if list is empty
            if self.list_files.count() == 0:
                placeholder_item = QtWidgets.QListWidgetItem("Drag and drop audio files here...")
                placeholder_item.setFlags(QtCore.Qt.ItemFlag.NoItemFlags)
                self.list_files.addItem(placeholder_item)
    
    def load_config(self):
        """Load configuration from file"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                
                # Restore UI values
                self.sb_m.setValue(config.get('m', 8))
                self.dsb_decay.setValue(config.get('decay', 1.25))
                self.sb_pre.setValue(config.get('pre_hp', 3000))
                self.sb_post.setValue(config.get('post_hp', 16000))
                self.sb_order.setValue(config.get('filter_order', 11))
                self.sb_sr.setValue(config.get('target_sr', 96000))
                self.le_outdir.setText(config.get('output_dir', os.path.abspath("output")))
                
                # Set format index based on saved format
                format_map = {'ALAC': 0, 'FLAC': 1, 'MP3': 2}
                format_index = format_map.get(config.get('format', 'ALAC'), 0)
                self.cb_format.setCurrentIndex(format_index)
                
                # Load recent files
                self.recent_files = config.get('recent_files', [])
                self.update_recent_files_menu()
                
                # Load dark mode setting
                self.dark_mode = config.get('dark_mode', False)
                self.apply_theme()
                
                # Update button text to reflect loaded state
                if self.dark_mode:
                    self.btn_test_dark.setText("Light Mode")
                else:
                    self.btn_test_dark.setText("Dark Mode")
                
        except Exception as e:
            self.append_log(f"Failed to load config: {e}")
    
    def save_config(self):
        """Save current configuration to file"""
        try:
            config = {
                'm': self.sb_m.value(),
                'decay': self.dsb_decay.value(),
                'pre_hp': self.sb_pre.value(),
                'post_hp': self.sb_post.value(),
                'filter_order': self.sb_order.value(),
                'target_sr': self.sb_sr.value(),
                'output_dir': self.le_outdir.text(),
                'format': self.cb_format.currentText(),
                'recent_files': self.recent_files,
                'dark_mode': self.dark_mode
            }
            
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
                
        except Exception as e:
            self.append_log(f"Failed to save config: {e}")
    
    def add_to_recent_files(self, file_path: str):
        """Add file to recent files list"""
        if file_path in self.recent_files:
            self.recent_files.remove(file_path)
        self.recent_files.insert(0, file_path)
        
        # Keep only max_recent_files
        if len(self.recent_files) > self.max_recent_files:
            self.recent_files = self.recent_files[:self.max_recent_files]
        
        # Update recent files menu
        self.update_recent_files_menu()
    
    def update_recent_files_menu(self):
        """Update the recent files menu"""
        self.recent_menu.clear()
        
        if not self.recent_files:
            action = QAction("No recent files", self)
            action.setEnabled(False)
            self.recent_menu.addAction(action)
        else:
            for file_path in self.recent_files:
                action = QAction(os.path.basename(file_path), self)
                action.setToolTip(file_path)
                action.triggered.connect(lambda checked, path=file_path: self.load_recent_file(path))
                self.recent_menu.addAction(action)
    
    def load_recent_file(self, file_path: str):
        """Load a recent file"""
        if os.path.exists(file_path):
            # Remove placeholder if it exists
            if self.list_files.count() == 1 and self.list_files.item(0).flags() == QtCore.Qt.ItemFlag.NoItemFlags:
                self.list_files.takeItem(0)
            
            # Add file if not already in list
            if not self.list_files.findItems(file_path, QtCore.Qt.MatchFlag.MatchExactly):
                self.list_files.addItem(file_path)
        else:
            # Remove from recent files if file no longer exists
            if file_path in self.recent_files:
                self.recent_files.remove(file_path)
                self.update_recent_files_menu()
            QtWidgets.QMessageBox.warning(self, "File Not Found", f"The file {file_path} no longer exists.")

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
        self.worker.sig_processing_stats.connect(self.on_processing_stats)
        self.worker.sig_retry_available.connect(self.on_retry_available)
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

    @QtCore.Slot(str, str)
    def on_retry_available(self, fname, err):
        """Handle when a file fails and retry is available"""
        self.failed_files.append(fname)
        self.btn_retry.setEnabled(True)
        self.append_log(f"[Retry Available] {fname}: {err}")

    def on_retry_failed(self):
        """Retry processing failed files"""
        if not self.failed_files:
            return
        
        self.append_log(f"Retrying {len(self.failed_files)} failed files...")
        
        # Reset progress
        self.pb_all.setValue(0)
        self.pb_file.setValue(0)
        self.lbl_now.setText("Retrying failed files...")
        
        # Lock buttons
        self.btn_start.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.btn_retry.setEnabled(False)
        
        # Start background thread with failed files
        self.worker = DSREWorker(self.failed_files, self.le_outdir.text().strip() or os.path.abspath("output"), self.params())
        self.worker.sig_log.connect(self.append_log)
        self.worker.sig_file_progress.connect(self.on_file_progress)
        self.worker.sig_step_progress.connect(self.on_step_progress)
        self.worker.sig_overall_progress.connect(self.on_overall_progress)
        self.worker.sig_file_done.connect(self.on_file_done)
        self.worker.sig_error.connect(self.on_error)
        self.worker.sig_finished.connect(self.on_retry_finished)
        self.worker.sig_processing_stats.connect(self.on_processing_stats)
        self.worker.sig_retry_available.connect(self.on_retry_available)
        self.worker.start()

    def on_retry_finished(self):
        """Handle retry completion"""
        self.append_log("Retry processing completed")
        self.lbl_now.setText("Control")
        self.btn_start.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.btn_retry.setEnabled(len(self.failed_files) > 0)
        self.worker = None

    @QtCore.Slot(dict)
    def on_processing_stats(self, stats):
        """Update processing statistics display"""
        if stats['start_time']:
            elapsed = time.time() - stats['start_time']
            processed = stats['processed_files']
            total = stats['total_files']
            
            if processed > 0:
                # Calculate ETA
                avg_time_per_file = elapsed / processed
                remaining_files = total - processed
                eta_seconds = remaining_files * avg_time_per_file
                
                # Format time
                eta_str = self.format_time(eta_seconds)
                elapsed_str = self.format_time(elapsed)
                
                # Update statistics
                stats_text = f"Processed: {processed}/{total} files"
                if stats['total_size_mb'] > 0:
                    processed_mb = stats['processed_size_mb']
                    total_mb = stats['total_size_mb']
                    stats_text += f" | {processed_mb:.1f}/{total_mb:.1f} MB"
                
                self.lbl_stats.setText(stats_text)
                self.lbl_eta.setText(f"Elapsed: {elapsed_str} | ETA: {eta_str}")
            else:
                self.lbl_stats.setText(f"Starting processing of {total} files...")
                self.lbl_eta.setText("")
        else:
            self.lbl_stats.setText("Ready to process")
            self.lbl_eta.setText("")
    
    def format_time(self, seconds):
        """Format time in seconds to human readable format"""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds/60:.0f}m {seconds%60:.0f}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"

    def on_cancel(self):
        if self.worker and self.worker.isRunning():
            self.append_log("Cancelling...")
            self.worker.abort()

    def on_finished(self):
        self.append_log("All files have been processed")
        self.lbl_now.setText("Control")
        self.btn_start.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.btn_retry.setEnabled(len(self.failed_files) > 0)
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
