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

    # Ensure shape is (samples, channels) for soundfile
    if y_out.ndim == 1:
        data = y_out[:, None]  # Add channel dimension
    else:
        # Convert from (channels, samples) to (samples, channels)
        data = y_out.T if y_out.shape[0] < y_out.shape[1] else y_out

    # Convert to float32 and normalize
    data = data.astype(np.float32, copy=False)
    
    # Debug: Log data properties before saving
    print(f"DEBUG: Audio data shape: {data.shape}, dtype: {data.dtype}")
    print(f"DEBUG: Audio data range: {np.min(data):.6f} to {np.max(data):.6f}")
    print(f"DEBUG: Audio data RMS: {np.sqrt(np.mean(data**2)):.6f}")
    
    # Check for NaN values
    if np.any(np.isnan(data)):
        print(f"ERROR: Audio data contains NaN values! This will cause silent output.")
        print(f"DEBUG: NaN count: {np.sum(np.isnan(data))} out of {data.size} samples")
        raise ValueError("Audio data contains NaN values - cannot save")
    
    # Check for infinite values
    if np.any(np.isinf(data)):
        print(f"ERROR: Audio data contains infinite values!")
        print(f"DEBUG: Inf count: {np.sum(np.isinf(data))} out of {data.size} samples")
        raise ValueError("Audio data contains infinite values - cannot save")
    
    if normalize:
        peak = float(np.max(np.abs(data)))
        if peak > 1.0:
            data /= peak
    else:
        data = np.clip(data, -1.0, 1.0)
    
    # Ensure we have valid audio data
    if np.max(np.abs(data)) < 1e-10:
        raise ValueError("Audio data is essentially silent - cannot save")

    # Temporary WAV file with proper cleanup
    tmp_wav = None
    cover_tmp = None
    try:
        tmp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp_wav.close()
        
        # Write temporary WAV file
        sf.write(tmp_wav.name, data, sr, subtype="FLOAT")
        
        # Verify the temporary file was written correctly
        if not os.path.exists(tmp_wav.name):
            raise Exception("Failed to create temporary WAV file")
        
        # Check file size
        file_size = os.path.getsize(tmp_wav.name)
        if file_size < 1000:  # Less than 1KB is suspicious
            raise Exception(f"Temporary WAV file is too small ({file_size} bytes) - audio data may be invalid")
        
        print(f"DEBUG: Created temporary WAV file: {tmp_wav.name} ({file_size} bytes)")

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
            print(f"DEBUG: Running FFmpeg command: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"DEBUG: FFmpeg completed successfully")
            
            # Verify output file was created and has reasonable size
            if not os.path.exists(out_path):
                raise Exception(f"Output file was not created: {out_path}")
            
            output_size = os.path.getsize(out_path)
            if output_size < 1000:  # Less than 1KB is suspicious
                raise Exception(f"Output file is too small ({output_size} bytes) - may be silent")
            
            print(f"DEBUG: Output file created successfully: {out_path} ({output_size} bytes)")
            
        except subprocess.CalledProcessError as e:
            print(f"DEBUG: FFmpeg stderr: {e.stderr}")
            print(f"DEBUG: FFmpeg stdout: {e.stdout}")
            raise Exception(f"FFmpeg command failed: {' '.join(cmd)}. Error: {e.stderr}")
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

# ======== ENHANCED AUDIO PROCESSING ALGORITHMS ========

def generate_harmonics(signal_band, fundamental_freq, sr, num_harmonics=5, harmonic_strength=0.3):
    """Generate harmonic content for a frequency band"""
    if len(signal_band) == 0:
        return signal_band
    
    # Check for NaN values in input
    if np.any(np.isnan(signal_band)):
        print("WARNING: NaN values in input to generate_harmonics, returning original")
        return signal_band
    
    # Create harmonic series
    enhanced = signal_band.copy()
    
    for h in range(2, num_harmonics + 2):  # 2nd, 3rd, 4th, 5th harmonics
        harmonic_freq = fundamental_freq * h
        if harmonic_freq < sr / 2:  # Below Nyquist
            # Create harmonic by frequency shifting
            phase_increment = 2 * np.pi * harmonic_freq / sr
            
            # Check for invalid phase increment
            if np.isnan(phase_increment) or np.isinf(phase_increment):
                print(f"WARNING: Invalid phase increment for harmonic {h}, skipping...")
                continue
            
            harmonic_oscillator = np.sin(phase_increment * np.arange(len(signal_band)))
            
            # Check for NaN values in oscillator
            if np.any(np.isnan(harmonic_oscillator)):
                print(f"WARNING: NaN values in harmonic oscillator {h}, skipping...")
                continue
            
            # Modulate the original signal to create harmonic content
            harmonic_content = signal_band * harmonic_oscillator * (harmonic_strength / h)
            
            # Check for NaN values in harmonic content
            if np.any(np.isnan(harmonic_content)):
                print(f"WARNING: NaN values in harmonic content {h}, skipping...")
                continue
            
            enhanced += harmonic_content
    
    # Final check for NaN values
    if np.any(np.isnan(enhanced)):
        print("WARNING: NaN values in enhanced signal, returning original")
        return signal_band
    
    return enhanced

def multiband_exciter(x, sr, progress_cb=None, abort_cb=None):
    """
    Multi-band harmonic exciter that adds pleasant harmonics and presence
    """
    if x.ndim == 1:
        x = x[np.newaxis, :]
    
    enhanced = np.zeros_like(x)
    
    # Define frequency bands for enhancement with better frequency range validation
    nyquist = sr // 2
    bands = []
    
    # Only add bands that are well within the valid frequency range
    band_definitions = [
        {"name": "Sub Bass", "low": 20, "high": 80, "gain": 1.2, "harmonics": 3, "strength": 0.15},
        {"name": "Bass", "low": 80, "high": 250, "gain": 1.4, "harmonics": 4, "strength": 0.25},
        {"name": "Low Mid", "low": 250, "high": 800, "gain": 1.6, "harmonics": 5, "strength": 0.35},
        {"name": "Mid", "low": 800, "high": 2500, "gain": 1.8, "harmonics": 6, "strength": 0.4},
        {"name": "High Mid", "low": 2500, "high": 8000, "gain": 2.2, "harmonics": 4, "strength": 0.45},
        {"name": "Presence", "low": 8000, "high": 16000, "gain": 2.8, "harmonics": 3, "strength": 0.3},
        {"name": "Air", "low": 16000, "high": min(20000, nyquist - 1000), "gain": 3.5, "harmonics": 2, "strength": 0.2}
    ]
    
    # Filter bands to only include those that are valid for the current sample rate
    for band in band_definitions:
        if band["low"] < nyquist and band["high"] < nyquist and band["high"] > band["low"]:
            bands.append(band)
        else:
            print(f"INFO: Skipping band {band['name']} due to sample rate limitations (nyquist={nyquist}Hz)")
    
    if not bands:
        print("WARNING: No valid frequency bands for current sample rate, using original signal")
        return x
    
    for ch in range(x.shape[0]):
        if abort_cb and abort_cb():
            break
            
        # Start with original signal instead of zeros
        channel_enhanced = x[ch].copy()
        
        for i, band in enumerate(bands):
            if abort_cb and abort_cb():
                break
                
            if progress_cb:
                progress = int((i + ch * len(bands)) * 100 / (len(bands) * x.shape[0]))
                progress_cb(progress, f"Processing band {band['name']}")
            
            # Skip if band exceeds Nyquist
            if band["low"] >= sr // 2:
                continue
                
            # Design bandpass filter with better parameter validation
            low_norm = band["low"] / (sr / 2)
            high_norm = min(band["high"] / (sr / 2), 0.99)
            
            # Debug: Log frequency range information
            print(f"DEBUG: Band {band['name']}: {band['low']}-{band['high']}Hz -> {low_norm:.4f}-{high_norm:.4f} (nyquist={sr//2}Hz)")
            
            # Ensure valid frequency range
            if low_norm >= high_norm or low_norm <= 0 or high_norm >= 1.0:
                print(f"WARNING: Invalid frequency range for band {band['name']} ({low_norm:.4f}-{high_norm:.4f}), skipping...")
                continue
            
            # Ensure minimum frequency separation (very lenient for low frequencies)
            if low_norm < 0.01:  # Very low frequencies (below 1% of Nyquist)
                min_separation = 0.0001  # Very lenient
            elif low_norm < 0.1:  # Low frequencies (below 10% of Nyquist)
                min_separation = 0.001   # Lenient
            else:
                min_separation = 0.01    # Standard
                
            if high_norm - low_norm < min_separation:
                print(f"WARNING: Frequency range too narrow for band {band['name']} ({low_norm:.4f}-{high_norm:.4f}), skipping...")
                continue
                
            try:
                # Use lower order filter for better stability
                filter_order = min(4, max(2, int(4 * (high_norm - low_norm))))
                b, a = signal.butter(filter_order, [low_norm, high_norm], btype='band')
                
                # Check for invalid filter coefficients
                if np.any(np.isnan(b)) or np.any(np.isnan(a)) or np.any(np.isinf(b)) or np.any(np.isinf(a)):
                    print(f"WARNING: Invalid filter coefficients for band {band['name']}, skipping...")
                    continue
                
                # Apply filter with error handling
                try:
                    band_signal = signal.filtfilt(b, a, x[ch])
                except Exception as filter_error:
                    print(f"WARNING: Filter application failed for band {band['name']}: {filter_error}, skipping...")
                    continue
                
                # Check for NaN values after filtering
                if np.any(np.isnan(band_signal)):
                    print(f"WARNING: NaN values after filtering in band {band['name']}, skipping...")
                    continue
                
                # Add harmonic excitement
                center_freq = (band["low"] + band["high"]) / 2
                harmonics_added = generate_harmonics(
                    band_signal, center_freq, sr, 
                    band["harmonics"], band["strength"]
                )
                
                # Check for NaN values after harmonic generation
                if np.any(np.isnan(harmonics_added)):
                    print(f"WARNING: NaN values after harmonic generation in band {band['name']}, skipping...")
                    continue
                
                # Apply gentle saturation for warmth
                saturated = np.tanh(harmonics_added * 1.5) * 0.8
                
                # Check for NaN values after saturation
                if np.any(np.isnan(saturated)):
                    print(f"WARNING: NaN values after saturation in band {band['name']}, skipping...")
                    continue
                
                # Apply band gain and blend with original
                band_enhanced = saturated * band["gain"]
                
                # Check for NaN values in band processing
                if np.any(np.isnan(band_enhanced)):
                    print(f"WARNING: NaN values detected in band {band['name']} processing, skipping...")
                    continue
                
                # Blend the enhanced band with the channel (subtle enhancement)
                channel_enhanced = channel_enhanced + band_enhanced * 0.3
                
            except Exception as e:
                # Skip problematic bands but continue processing
                continue
        
        enhanced[ch] = channel_enhanced
    
    return enhanced

def psychoacoustic_enhancer(x, sr, progress_cb=None, abort_cb=None):
    """
    Psychoacoustic enhancement targeting human hearing sensitivity
    """
    if x.ndim == 1:
        x = x[np.newaxis, :]
    
    enhanced = np.zeros_like(x)
    
    # A-weighting inspired frequency response (emphasizes 2-5kHz)
    critical_bands = [
        {"freq": 1000, "boost": 1.8, "q": 1.5},   # Fundamental vocal range
        {"freq": 2500, "boost": 2.5, "q": 2.0},   # Presence and clarity
        {"freq": 4000, "boost": 3.0, "q": 1.8},   # Maximum hearing sensitivity
        {"freq": 6000, "boost": 2.2, "q": 1.2},   # Consonant definition
        {"freq": 10000, "boost": 1.6, "q": 0.8},  # Air and sparkle
    ]
    
    for ch in range(x.shape[0]):
        if abort_cb and abort_cb():
            break
            
        channel_enhanced = x[ch].copy()
        
        for i, band in enumerate(critical_bands):
            if abort_cb and abort_cb():
                break
                
            if progress_cb:
                progress = int((i + ch * len(critical_bands)) * 100 / (len(critical_bands) * x.shape[0]))
                progress_cb(progress, f"Psychoacoustic enhancement at {band['freq']}Hz")
            
            # Skip if frequency exceeds Nyquist
            if band["freq"] >= sr // 2:
                continue
            
            try:
                # Create bell filter (peaking EQ)
                freq_norm = band["freq"] / (sr / 2)
                if freq_norm >= 0.99:
                    continue
                    
                # Design peaking filter
                # Using a simple approach since scipy doesn't have direct peaking filter
                w = 2 * np.pi * band["freq"] / sr
                cosw = np.cos(w)
                sinw = np.sin(w)
                alpha = sinw / (2 * band["q"])
                A = 10**(band["boost"]/40)  # Convert dB to linear
                
                # Peaking filter coefficients
                b0 = 1 + alpha * A
                b1 = -2 * cosw
                b2 = 1 - alpha * A
                a0 = 1 + alpha / A
                a1 = -2 * cosw
                a2 = 1 - alpha / A
                
                # Normalize
                b = np.array([b0, b1, b2]) / a0
                a = np.array([1, a1/a0, a2/a0])
                
                # Apply filter
                filtered = signal.lfilter(b, a, x[ch])
                
                # Blend with original (subtle enhancement)
                blend_factor = 0.4
                channel_enhanced = channel_enhanced * (1 - blend_factor) + filtered * blend_factor
                
            except Exception as e:
                continue
        
        enhanced[ch] = channel_enhanced
    
    return enhanced

def stereo_width_enhancer(x, width_factor=1.4):
    """
    Enhance stereo width using M/S processing
    """
    if x.shape[0] != 2:  # Only works on stereo
        return x
    
    left = x[0]
    right = x[1]
    
    # Convert to Mid/Side
    mid = (left + right) / 2
    side = (left - right) / 2
    
    # Enhance side channel for wider stereo image
    side_enhanced = side * width_factor
    
    # Convert back to L/R
    left_enhanced = mid + side_enhanced
    right_enhanced = mid - side_enhanced
    
    return np.array([left_enhanced, right_enhanced])

def dynamic_range_enhancer(x, ratio=1.3, attack_ms=5, release_ms=50, sr=44100):
    """
    Gentle upward expansion to increase dynamic range and liveliness
    """
    # Convert ms to samples
    attack_samples = int(attack_ms * sr / 1000)
    release_samples = int(release_ms * sr / 1000)
    
    enhanced = np.zeros_like(x)
    
    for ch in range(x.shape[0]):
        signal_ch = x[ch]
        
        # Calculate envelope
        envelope = np.abs(signal_ch)
        
        # Smooth envelope
        if len(envelope) > 0:
            # Simple envelope follower
            smoothed_env = np.zeros_like(envelope)
            current_env = envelope[0]
            
            for i in range(len(envelope)):
                if envelope[i] > current_env:
                    # Attack
                    current_env += (envelope[i] - current_env) / attack_samples
                else:
                    # Release
                    current_env -= (current_env - envelope[i]) / release_samples
                
                smoothed_env[i] = current_env
            
            # Apply upward expansion
            threshold = 0.1  # -20dB
            gain = np.ones_like(smoothed_env)
            
            # Only expand signals above threshold
            above_threshold = smoothed_env > threshold
            gain[above_threshold] = (smoothed_env[above_threshold] / threshold) ** (ratio - 1)
            
            # Limit maximum gain
            gain = np.clip(gain, 1.0, 3.0)
            
            enhanced[ch] = signal_ch * gain
    
    return enhanced

def enhanced_audio_algorithm(
    x: np.ndarray,
    sr: int,
    enhancement_strength: float = 0.7,
    harmonic_intensity: float = 0.6,
    stereo_width: float = 1.3,
    dynamic_enhancement: float = 1.2,
    progress_cb: Optional[callable] = None,
    abort_cb: Optional[callable] = None,
) -> np.ndarray:
    """
    Complete enhanced audio processing algorithm
    
    Args:
        x: Input audio data (channels, samples)
        sr: Sample rate in Hz
        enhancement_strength: Overall enhancement strength (0.1-1.0)
        harmonic_intensity: Harmonic generation intensity (0.1-1.0)
        stereo_width: Stereo width enhancement (1.0-2.0)
        dynamic_enhancement: Dynamic range enhancement (1.0-2.0)
        progress_cb: Optional callback for progress updates
        abort_cb: Optional callback to check for abort signal
        
    Returns:
        Enhanced audio data with same shape as input
    """
    # Input validation
    if x is None or x.size == 0:
        raise ValueError("Input audio data is empty or None")
    
    if np.max(np.abs(x)) < 1e-10:
        raise ValueError("Input audio data appears to be silent")
    
    # Check for NaN values in input
    if np.any(np.isnan(x)):
        raise ValueError("Input audio data contains NaN values")
    
    # Check for infinite values in input
    if np.any(np.isinf(x)):
        raise ValueError("Input audio data contains infinite values")
    
    if progress_cb:
        progress_cb(0, "Starting enhancement process")
    
    # Step 1: Multi-band harmonic excitement
    if progress_cb:
        progress_cb(10, "Applying multi-band harmonic excitement")
    
    enhanced = multiband_exciter(x, sr, 
                               lambda p, desc: progress_cb(10 + p//4, desc) if progress_cb else None,
                               abort_cb)
    
    # Check for NaN values after multiband processing
    if np.any(np.isnan(enhanced)):
        print("ERROR: NaN values detected after multiband processing!")
        print("INFO: Falling back to simple enhancement approach...")
        
        # Fallback: Simple gentle enhancement without complex filtering
        enhanced = x.copy()
        for ch in range(x.shape[0]):
            # Simple gentle saturation and harmonic enhancement
            signal_ch = x[ch]
            
            # Add gentle harmonic content
            enhanced_ch = signal_ch.copy()
            for harmonic in [2, 3, 4]:
                if harmonic * 1000 < sr // 2:  # Ensure harmonic is below Nyquist
                    phase = 2 * np.pi * harmonic * 1000 / sr * np.arange(len(signal_ch))
                    harmonic_content = signal_ch * np.sin(phase) * 0.1
                    enhanced_ch += harmonic_content
            
            # Gentle saturation
            enhanced_ch = np.tanh(enhanced_ch * 1.2) * 0.9
            
            # Blend with original
            enhanced[ch] = signal_ch * 0.7 + enhanced_ch * 0.3
    
    if abort_cb and abort_cb():
        return x
    
    # Step 2: Psychoacoustic enhancement
    if progress_cb:
        progress_cb(35, "Applying psychoacoustic enhancement")
    
    psycho_enhanced = psychoacoustic_enhancer(enhanced, sr,
                                            lambda p, desc: progress_cb(35 + p//4, desc) if progress_cb else None,
                                            abort_cb)
    
    # Check for NaN values after psychoacoustic processing
    if np.any(np.isnan(psycho_enhanced)):
        print("ERROR: NaN values detected after psychoacoustic processing!")
        return x  # Return original audio
    
    if abort_cb and abort_cb():
        return x
    
    # Step 3: Dynamic range enhancement
    if progress_cb:
        progress_cb(60, "Enhancing dynamic range")
    
    dynamic_enhanced = dynamic_range_enhancer(psycho_enhanced, dynamic_enhancement, sr=sr)
    
    # Check for NaN values after dynamic range processing
    if np.any(np.isnan(dynamic_enhanced)):
        print("ERROR: NaN values detected after dynamic range processing!")
        return x  # Return original audio
    
    if abort_cb and abort_cb():
        return x
    
    # Step 4: Stereo width enhancement (if stereo)
    if progress_cb:
        progress_cb(75, "Enhancing stereo width")
    
    if x.shape[0] == 2:
        stereo_enhanced = stereo_width_enhancer(dynamic_enhanced, stereo_width)
    else:
        stereo_enhanced = dynamic_enhanced
    
    if abort_cb and abort_cb():
        return x
    
    # Step 5: Final blend and normalization
    if progress_cb:
        progress_cb(90, "Final processing and normalization")
    
    # Ensure we have valid audio data
    if np.max(np.abs(stereo_enhanced)) < 1e-10:
        # If enhanced signal is essentially silent, return original
        final = x.copy()
    else:
        # Blend enhanced with original - ensure we don't lose the original signal
        blend_factor = min(enhancement_strength, 0.8)  # Cap at 80% to preserve original
        final = x * (1 - blend_factor) + stereo_enhanced * blend_factor
    
    # Gentle limiting to prevent clipping
    peak = np.max(np.abs(final))
    if peak > 0.95:
        final = final * (0.95 / peak)
    
    # Ensure we have some audio content
    if np.max(np.abs(final)) < 1e-10:
        # If final result is silent, return original
        final = x.copy()
    
    if progress_cb:
        progress_cb(100, "Enhancement complete")
    
    return final


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
            return enhanced_audio_algorithm(
                y, sr,
                enhancement_strength=float(self.params["decay"]),
                harmonic_intensity=float(self.params["m"]) / 16.0,
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
                processed_chunk = enhanced_audio_algorithm(
                    chunk, sr,
                    enhancement_strength=float(self.params["decay"]),
                    harmonic_intensity=float(self.params["m"]) / 16.0,
                    progress_cb=None,
                    abort_cb=lambda: self._abort
                )
                chunks.append(processed_chunk)
        
        return np.concatenate(chunks) if chunks else y
    
    def check_audio_file_format(self, file_path: str) -> bool:
        """Check if the audio file format is supported"""
        try:
            import soundfile as sf
            with sf.SoundFile(file_path) as f:
                # Just check if we can open the file
                return True
        except Exception:
            return False
    
    def load_audio_with_recovery(self, file_path: str) -> Tuple[np.ndarray, int]:
        """Load audio with multiple fallback strategies for corrupted files"""
        import warnings
        
        # Check file format first
        if not self.check_audio_file_format(file_path):
            self.sig_log.emit(f"WARNING: Audio file format may not be fully supported: {os.path.basename(file_path)}")
        
        # Suppress librosa warnings for cleaner output
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="librosa")
            warnings.filterwarnings("ignore", category=FutureWarning, module="librosa")
            
            try:
                # First attempt: Normal loading with explicit format detection
                self.sig_log.emit(f"Loading audio: {os.path.basename(file_path)}")
                y, sr = librosa.load(file_path, mono=False, sr=None)
                
                # Validate loaded audio
                if y is None or y.size == 0:
                    raise ValueError("Empty audio data loaded")
                
                self.sig_log.emit(f"Successfully loaded: {y.shape}, {sr}Hz")
                return y, sr
                
            except Exception as e1:
                self.sig_log.emit(f"Primary load failed: {str(e1)[:100]}...")
                
                try:
                    # Second attempt: Force mono and resample
                    self.sig_log.emit(f"Attempting mono recovery...")
                    y, sr = librosa.load(file_path, mono=True, sr=None)
                    
                    if y is None or y.size == 0:
                        raise ValueError("Empty audio data loaded")
                    
                    self.sig_log.emit(f"Mono recovery successful: {y.shape}, {sr}Hz")
                    return y, sr
                    
                except Exception as e2:
                    try:
                        # Third attempt: Use soundfile directly with error handling
                        self.sig_log.emit(f"Attempting direct soundfile loading...")
                        import soundfile as sf
                        y, sr = sf.read(file_path, always_2d=True)
                        
                        if y is None or y.size == 0:
                            raise ValueError("Empty audio data loaded")
                        
                        if y.ndim == 1:
                            y = y[:, np.newaxis]
                        
                        self.sig_log.emit(f"Soundfile loading successful: {y.shape}, {sr}Hz")
                        return y.T, sr  # Convert to (channels, samples)
                        
                    except Exception as e3:
                        try:
                            # Fourth attempt: Load with different parameters
                            self.sig_log.emit(f"Attempting alternative loading parameters...")
                            y, sr = librosa.load(file_path, mono=False, sr=44100)  # Force 44.1kHz
                            
                            if y is None or y.size == 0:
                                raise ValueError("Empty audio data loaded")
                            
                            self.sig_log.emit(f"Alternative loading successful: {y.shape}, {sr}Hz")
                            return y, sr
                            
                        except Exception as e4:
                            # Final attempt: Raise error instead of creating silence
                            self.sig_log.emit(f"All recovery methods failed for {os.path.basename(file_path)}")
                            self.sig_log.emit(f"Error details: {str(e1)[:50]}, {str(e2)[:50]}, {str(e3)[:50]}, {str(e4)[:50]}")
                            raise Exception(f"Unable to load audio file {os.path.basename(file_path)}. All recovery methods failed.")
    
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

                    # Process with new enhanced algorithm
                    def step_cb(cur, desc):
                        pct = int(cur)
                        self.sig_step_progress.emit(pct, fname)

                    # Debug: Log input audio properties
                    self.sig_log.emit(f"Input audio shape: {y.shape}, sample rate: {sr} Hz")
                    self.sig_log.emit(f"Input audio range: {np.min(y):.4f} to {np.max(y):.4f}")
                    self.sig_log.emit(f"Input audio RMS: {np.sqrt(np.mean(y**2)):.6f}")
                    self.sig_log.emit(f"Enhancement parameters: strength={self.params['decay']}, harmonics={self.params['m']}")
                    
                    # Check if input audio is essentially silent
                    if np.max(np.abs(y)) < 1e-10:
                        self.sig_log.emit(f"WARNING: Input audio appears to be silent!")
                        raise Exception("Input audio file appears to be silent or corrupted")

                    # Use chunked processing for files > 50MB
                    if file_size_mb > 50:
                        self.sig_log.emit(f"Using chunked processing for large file: {fname}")
                        y_out = self.process_audio_chunked(y, sr)
                    else:
                        self.sig_log.emit(f"Starting Enhanced Audio Processing...")
                        y_out = enhanced_audio_algorithm(
                            y, sr,
                            enhancement_strength=float(self.params["decay"]),
                            harmonic_intensity=float(self.params["m"]) / 16.0,
                            stereo_width=1.3,
                            dynamic_enhancement=1.2,
                            progress_cb=step_cb,
                            abort_cb=lambda: self._abort
                        )
                    
                    # Debug: Log output audio properties
                    self.sig_log.emit(f"Output audio shape: {y_out.shape}, sample rate: {sr} Hz")
                    self.sig_log.emit(f"Output audio range: {np.min(y_out):.4f} to {np.max(y_out):.4f}")
                    self.sig_log.emit(f"Output audio RMS: {np.sqrt(np.mean(y_out**2)):.6f}")
                    
                    # Check if output audio is essentially silent
                    if np.max(np.abs(y_out)) < 1e-10:
                        self.sig_log.emit(f"ERROR: Output audio is silent! Using original audio instead.")
                        y_out = y.copy()
                    
                    # Calculate and log the difference
                    if y.shape == y_out.shape:
                        diff = np.abs(y_out - y)
                        max_diff = np.max(diff)
                        mean_diff = np.mean(diff)
                        rms_original = np.sqrt(np.mean(y**2))
                        rms_enhanced = np.sqrt(np.mean(y_out**2))
                        enhancement_ratio = rms_enhanced / (rms_original + 1e-12)
                        
                        self.sig_log.emit(f"Enhancement Results:")
                        self.sig_log.emit(f"  Max difference: {max_diff:.6f}")
                        self.sig_log.emit(f"  Mean difference: {mean_diff:.6f}")
                        self.sig_log.emit(f"  RMS enhancement ratio: {enhancement_ratio:.3f}")
                        
                        if max_diff < 0.001:
                            self.sig_log.emit(f"WARNING: Very small enhancement detected!")
                        else:
                            self.sig_log.emit(f"SUCCESS: Significant audio enhancement applied!")

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
                    out_path = os.path.join(self.output_dir, f"{base}_enhanced.{ext}")
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
        self.setWindowTitle("DSRE v2.0.Enhanced - Audio Enhancement Suite")

        # Get relative path icon
        icon_path = os.path.join(os.path.dirname(__file__), "logo.ico")
        self.setWindowIcon(QIcon(icon_path))

        self.resize(1200, 800)
        
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
        self.status_bar.showMessage("Ready - Enhanced Audio Processing Algorithm Loaded")

        # File list with drag & drop support
        self.list_files = DragDropListWidget()
        self.list_files.setToolTip("Drag and drop audio files here for enhancement processing")
        
        # Ensure the widget can receive focus and events
        self.list_files.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.list_files.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        
        self.btn_add = QtWidgets.QPushButton("Add Input Files")
        self.btn_clear = QtWidgets.QPushButton("Clear Input List")
        self.btn_remove_selected = QtWidgets.QPushButton("Remove Selected")
        self.btn_outdir = QtWidgets.QPushButton("Select Output Directory")
        self.le_outdir = QtWidgets.QLineEdit()
        self.le_outdir.setPlaceholderText("Output folder")
        self.le_outdir.setText(os.path.abspath("enhanced_output"))

        # Enhanced Parameters with better descriptions
        self.sb_m = QtWidgets.QSpinBox()
        self.sb_m.setRange(1, 32)
        self.sb_m.setValue(16)  # Optimized for new algorithm
        self.sb_m.setToolTip("Harmonic intensity (1-32): Higher values add more harmonic richness")
        
        self.dsb_decay = QtWidgets.QDoubleSpinBox()
        self.dsb_decay.setRange(0.1, 1.0)
        self.dsb_decay.setSingleStep(0.05)
        self.dsb_decay.setValue(0.7)  # Enhancement strength
        self.dsb_decay.setToolTip("Enhancement strength (0.1-1.0): Controls overall enhancement intensity")
        
        
        self.sb_sr = QtWidgets.QSpinBox()
        self.sb_sr.setRange(44100, 192000)
        self.sb_sr.setSingleStep(22050)
        self.sb_sr.setValue(96000)
        self.sb_sr.setToolTip("Target sample rate: Higher rates preserve more frequency content")

        # Progress
        self.pb_file = QtWidgets.QProgressBar()    # Single file progress
        self.pb_all = QtWidgets.QProgressBar()     # Overall progress
        self.lbl_now = QtWidgets.QLabel("Control")
        
        # Processing statistics
        self.lbl_stats = QtWidgets.QLabel("Ready to process - Enhanced Algorithm Active")
        self.lbl_stats.setStyleSheet("QLabel { color: #666; font-size: 10px; }")
        self.lbl_eta = QtWidgets.QLabel("")
        self.lbl_eta.setStyleSheet("QLabel { color: #666; font-size: 10px; }")

        # Control buttons
        self.btn_start = QtWidgets.QPushButton("Start Enhanced Processing")
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
        lbl_files = QtWidgets.QLabel("Input Audio Files")
        lbl_files.setAlignment(QtCore.Qt.AlignHCenter)
        left_layout.addWidget(lbl_files)
        left_layout.addWidget(self.list_files)
        left_widget.setLayout(left_layout)
        main_splitter.addWidget(left_widget)
        
        # === Middle panel: Operations ===
        middle_widget = QtWidgets.QWidget()
        middle_layout = QtWidgets.QVBoxLayout()
        lbl_ops = QtWidgets.QLabel("Enhancement Operations")
        lbl_ops.setAlignment(QtCore.Qt.AlignHCenter)
        middle_layout.addWidget(lbl_ops)

        vbtn = QtWidgets.QVBoxLayout()
        vbtn.addWidget(self.btn_add)
        vbtn.addWidget(self.btn_clear)
        vbtn.addWidget(self.btn_remove_selected)
        vbtn.addSpacing(10)
        vbtn.addWidget(QtWidgets.QLabel("Enhanced Output Directory"))
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
        self.cb_format.setToolTip("Output format: ALAC (lossless), FLAC (lossless), MP3 (lossy)")
        vbtn.addWidget(QtWidgets.QLabel("Output Format"))
        vbtn.addWidget(self.cb_format)

        middle_layout.addLayout(vbtn)
        middle_widget.setLayout(middle_layout)
        main_splitter.addWidget(middle_widget)

        # === Right panel: Parameter settings + Progress ===
        right_widget = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout()
        lbl_params = QtWidgets.QLabel("Enhancement Parameters")
        lbl_params.setAlignment(QtCore.Qt.AlignHCenter)
        right_layout.addWidget(lbl_params)

        form = QtWidgets.QFormLayout()
        form.addRow("Harmonic Intensity (1-32):", self.sb_m)
        form.addRow("Enhancement Strength (0.1-1.0):", self.dsb_decay)
        form.addRow("Target Sample Rate (Hz):", self.sb_sr)
        right_layout.addLayout(form)

        right_layout.addSpacing(20)

        vprog = QtWidgets.QVBoxLayout()
        vprog.addWidget(QtWidgets.QLabel("Current File Enhancement Progress"))
        vprog.addWidget(self.pb_file)
        vprog.addWidget(QtWidgets.QLabel("Overall Processing Progress"))
        vprog.addWidget(self.pb_all)
        vprog.addWidget(self.lbl_stats)
        vprog.addWidget(self.lbl_eta)
        vprog.addStretch(1)
        right_layout.addLayout(vprog)
        right_widget.setLayout(right_layout)
        main_splitter.addWidget(right_widget)
        
        # Set initial sizes for panels (proportional)
        main_splitter.setSizes([300, 300, 400])
        
        # Create vertical splitter for main content and log
        vertical_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        vertical_splitter.addWidget(main_splitter)
        
        # === Bottom panel: Enhanced Processing Log ===
        log_widget = QtWidgets.QWidget()
        log_layout = QtWidgets.QVBoxLayout()
        log_layout.addWidget(QtWidgets.QLabel("Enhanced Processing Log"))
        log_layout.addWidget(self.te_log)
        log_widget.setLayout(log_layout)
        vertical_splitter.addWidget(log_widget)
        
        # Set initial sizes for vertical splitter
        vertical_splitter.setSizes([600, 200])
        
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
        
        # Connect list selection changes to update button states
        self.list_files.itemSelectionChanged.connect(self.update_button_states)

        self.worker: Optional[DSREWorker] = None
        self.config_file = os.path.join(os.path.dirname(__file__), "dsre_enhanced_config.json")
        self.failed_files = []  # Track failed files for retry
        
        # Load saved configuration
        self.load_config()
        
        # Connect parameter changes to auto-save
        self.sb_m.valueChanged.connect(self.save_config)
        self.dsb_decay.valueChanged.connect(self.save_config)
        self.sb_sr.valueChanged.connect(self.save_config)
        self.le_outdir.textChanged.connect(self.save_config)
        self.cb_format.currentTextChanged.connect(self.save_config)

        # Write enhanced welcome message after initialization
        self.append_log("DSRE v2.0 - Enhanced Audio Processing Suite")
        self.append_log("=" * 50)
        self.append_log("NEW FEATURES:")
        self.append_log("• Multi-band harmonic excitement for richer sound")
        self.append_log("• Psychoacoustic enhancement targeting human hearing")
        self.append_log("• Dynamic range enhancement for more liveliness")
        self.append_log("• Stereo width enhancement for immersive soundstage")
        self.append_log("• Intelligent frequency-dependent processing")
        self.append_log("=" * 50)
        self.append_log("Software by: Qu Le Fan (Enhanced by AI)")
        self.append_log("Feedback: Le_Fan_Qv@outlook.com")
        self.append_log("Discussion Group: 323861356 (QQ)")
        self.append_log("Ready for enhanced audio processing!")
        
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
        start_action = QAction('&Start Enhanced Processing', self)
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
        QtWidgets.QMessageBox.about(self, "About DSRE Enhanced", 
            "DSRE v2.0 - Enhanced Audio Processing Suite\n\n"
            "Enhanced with advanced multi-band harmonic excitement,\n"
            "psychoacoustic enhancement, and dynamic range processing.\n\n"
            "Features:\n"
            "• Multi-band harmonic generation for richer sound\n"
            "• Psychoacoustic enhancement targeting human hearing\n"
            "• Dynamic range enhancement for liveliness\n"
            "• Stereo width enhancement for immersive soundstage\n"
            "• Intelligent frequency-dependent processing\n\n"
            "Original Software by: Qu Le Fan\n"
            "Enhanced Algorithm by: AI Assistant\n"
            "Feedback: Le_Fan_Qv@outlook.com\n"
            "Discussion Group: 323861356 (QQ)")

    def on_add_files(self):
        filters = (
            "Audio Files (*.wav *.mp3 *.m4a *.flac *.ogg *.aiff *.aif *.aac *.wma *.mka);;"
            "All Files (*.*)"
        )
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "Select Audio Files for Enhancement", "", filters)
        
        # Remove placeholder if it exists
        if self.list_files.count() == 1 and self.list_files.item(0).flags() == QtCore.Qt.ItemFlag.NoItemFlags:
            self.list_files.takeItem(0)
        
        for f in files:
            if f and (self.list_files.findItems(f, QtCore.Qt.MatchFlag.MatchExactly) == []):
                self.list_files.addItem(f)
                # Add to recent files
                self.add_to_recent_files(f)
        
        # Update button states after adding files
        self.update_button_states()

    def on_choose_outdir(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Enhanced Output Directory", self.le_outdir.text() or "")
        if d:
            self.le_outdir.setText(d)
    
    def on_clear_files(self):
        """Clear all files from the list"""
        self.list_files.clear()
        self.lbl_stats.setText("Ready for enhanced processing")
        self.lbl_eta.setText("")
        self.pb_all.setValue(0)
        self.pb_file.setValue(0)
        self.te_log.clear()
        
        # Write enhanced welcome message after clearing
        self.append_log("DSRE v2.0 - Enhanced Audio Processing Suite")
        self.append_log("Ready for enhanced audio processing!")
        
        # Update button states after clearing
        self.update_button_states()
    
    def update_button_states(self):
        """Update button states based on current selection and list contents"""
        has_selection = len(self.list_files.selectedItems()) > 0
        has_items = self.list_files.count() > 0
        
        self.btn_remove_selected.setEnabled(has_selection)
        self.btn_retry.setEnabled(len(self.failed_files) > 0)
    
    def on_remove_selected(self):
        """Remove selected files from the list"""
        # Get all selected items
        selected_items = self.list_files.selectedItems()
        if not selected_items:
            return
        
        # Remove selected items (in reverse order to maintain indices)
        for item in reversed(selected_items):
            row = self.list_files.row(item)
            self.list_files.takeItem(row)
        
        # Update button states
        self.update_button_states()
    
    def load_config(self):
        """Load configuration from file"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                
                # Restore UI values with enhanced defaults
                self.sb_m.setValue(config.get('m', 16))
                self.dsb_decay.setValue(config.get('decay', 0.7))
                self.sb_sr.setValue(config.get('target_sr', 96000))
                self.le_outdir.setText(config.get('output_dir', os.path.abspath("enhanced_output")))
                
                # Set format index based on saved format
                format_map = {'ALAC': 0, 'FLAC': 1, 'MP3': 2}
                format_index = format_map.get(config.get('format', 'ALAC'), 0)
                self.cb_format.setCurrentIndex(format_index)
                
                # Load recent files
                self.recent_files = config.get('recent_files', [])
                self.update_recent_files_menu()
                
                # Load dark mode setting
                self.dark_mode = config.get('dark_mode', False)
                
        except Exception as e:
            self.append_log(f"Failed to load config: {e}")
    
    def save_config(self):
        """Save current configuration to file"""
        try:
            config = {
                'm': self.sb_m.value(),
                'decay': self.dsb_decay.value(),
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
            target_sr=self.sb_sr.value(),
            bit_depth=24,  # Fixed output 24bit
            format=self.cb_format.currentText()  # ALAC, FLAC, or MP3
        )

    def append_log(self, s: str):
        self.te_log.append(s)
        self.te_log.moveCursor(QTextCursor.End)

    def on_start(self):
        # Get all files, filtering out placeholder items
        files = []
        for i in range(self.list_files.count()):
            item = self.list_files.item(i)
            if item.flags() != QtCore.Qt.ItemFlag.NoItemFlags:  # Skip placeholder items
                files.append(item.text())
        
        if not files:
            QtWidgets.QMessageBox.warning(self, "No Files", "Please add at least one audio file for enhancement")
            return
        outdir = self.le_outdir.text().strip() or os.path.abspath("enhanced_output")

        # Reset progress
        self.pb_all.setValue(0)
        self.pb_file.setValue(0)
        self.lbl_now.setText("Initializing enhanced processing...")
        self.append_log(f"Starting enhanced processing of {len(files)} files...")
        self.append_log(f"Enhancement strength: {self.dsb_decay.value()}")
        self.append_log(f"Harmonic intensity: {self.sb_m.value()}")

        # Clear failed files list for new processing session
        self.failed_files.clear()

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
        self.lbl_now.setText(f"Enhanced Processing... [{cur}/{total}]: {fname}")
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
        self.append_log(f"Enhancement completed: {os.path.basename(in_path)} -> {out_path}")

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
        
        self.append_log(f"Retrying enhanced processing for {len(self.failed_files)} failed files...")
        
        # Reset progress
        self.pb_all.setValue(0)
        self.pb_file.setValue(0)
        self.lbl_now.setText("Retrying enhanced processing...")
        
        # Lock buttons
        self.btn_start.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.btn_retry.setEnabled(False)
        
        # Start background thread with failed files
        self.worker = DSREWorker(self.failed_files, self.le_outdir.text().strip() or os.path.abspath("enhanced_output"), self.params())
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
        self.append_log("Enhanced retry processing completed")
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
                stats_text = f"Enhanced: {processed}/{total} files"
                if stats['total_size_mb'] > 0:
                    processed_mb = stats['processed_size_mb']
                    total_mb = stats['total_size_mb']
                    stats_text += f" | {processed_mb:.1f}/{total_mb:.1f} MB"
                
                self.lbl_stats.setText(stats_text)
                self.lbl_eta.setText(f"Elapsed: {elapsed_str} | ETA: {eta_str}")
            else:
                self.lbl_stats.setText(f"Starting enhanced processing of {total} files...")
                self.lbl_eta.setText("")
        else:
            self.lbl_stats.setText("Ready for enhanced processing")
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
            self.append_log("Cancelling enhanced processing...")
            self.worker.abort()

    def on_finished(self):
        self.append_log("All files have been enhanced successfully!")
        self.append_log("Enhanced audio files saved with '_enhanced' suffix")
        self.lbl_now.setText("Control")
        self.btn_start.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.btn_retry.setEnabled(len(self.failed_files) > 0)
        self.worker = None

def main():
    import ctypes
    myappid = "com.lefanqv.dsre.enhanced"  # Enhanced application ID
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