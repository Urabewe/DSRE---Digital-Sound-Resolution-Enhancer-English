# DSRE / Deep Sound Resolution Enhancer

<img width="896" height="626" alt="image" src="https://github.com/user-attachments/assets/c7747e30-82b4-4804-a458-474d0d438c7a" />

## ✨ Enhancements & Improvements

This enhanced version of DSRE includes numerous improvements over the original implementation:

### 🎨 **User Interface Enhancements**
- **Dark Mode Support**: Toggle between light and dark themes with persistent settings
- **Drag & Drop Interface**: Intuitive file loading with visual feedback
- **Resizable Panels**: Customizable layout with splitter controls
- **Keyboard Shortcuts**: Quick access to common functions (F5, Escape, Ctrl+L, etc.)
- **Recent Files Menu**: Easy access to previously processed files
- **Enhanced File List**: Better selection handling and visual indicators
- **Status Bar**: Real-time processing information and feedback

### ⚡ **Performance Optimizations**
- **File Size-Based Progress**: Accurate progress estimation based on file sizes
- **Chunked Processing**: Memory-efficient processing for large files (>50MB)
- **Processing Statistics**: Real-time ETA and performance metrics
- **Background Processing**: Non-blocking UI with threaded audio processing

### 🛡️ **Error Recovery & Robustness**
- **Automatic Retry System**: Up to 3 retry attempts for failed operations
- **Multi-Level Audio Loading**: 5 fallback strategies for corrupted audio files
- **Intelligent Error Categorization**: Adaptive retry delays based on error types
- **Partial Processing Recovery**: Resume from where processing left off
- **Comprehensive Error Handling**: Detailed error messages and recovery suggestions

### 🎵 **Audio Processing Improvements**
- **MP3 Output Support**: High-quality MP3 encoding with libmp3lame
- **Enhanced Metadata Preservation**: Better cover art and metadata handling
- **Improved Sample Rate Handling**: More robust resampling and format conversion
- **Better Audio Loading**: Multiple fallback methods for various audio formats

### 💾 **Configuration & Persistence**
- **Settings Persistence**: All preferences saved automatically
- **Recent Files Tracking**: Remember recently processed files
- **Theme Persistence**: Dark/light mode preference saved
- **Parameter Auto-Save**: Real-time saving of all parameter changes

### 🔧 **Code Quality & Maintainability**
- **Comprehensive Type Hints**: Better code documentation and IDE support
- **Enhanced Documentation**: Detailed docstrings and code comments
- **Modular Architecture**: Clean separation of concerns
- **Error Logging**: Detailed logging for debugging and troubleshooting

### 🎯 **User Experience Improvements**
- **Intuitive Controls**: Clear button labels and tooltips
- **Visual Feedback**: Progress bars, status updates, and processing indicators
- **File Management**: Easy add/remove/clear operations for file lists
- **Output Organization**: Automatic output directory management
- **Processing Feedback**: Real-time updates on processing status and statistics

---

## Description

DSRE is a **high-performance audio enhancement tool** that can batch-convert any audio files into **high-resolution (Hi-Res) audio**.
Inspired by Sony DSEE HX, it uses a **non-deep-learning frequency enhancement algorithm**, allowing fast processing of large batches without heavy computation.

**Key Features:**

* **Batch Processing**: Convert multiple audio files at once.
* **Multiple Formats**: Supports WAV, MP3, FLAC, M4A, ALAC, etc.
* **Preserves Cover & Metadata**: No manual editing required.
* **Flexible Parameters**: Modulation count, decay, high-pass filter, etc.
* **Fast & Stable**: Does not rely on deep learning, fast processing.

---

## DSRE Installation Instructions

## Prerequisites
- Python 3.7 or higher
- Git
- Important: ffmpeg.exe must be located in a folder called ffmpeg within the project directory for the application to work properly

## Installation Steps

### 1. Clone the Repository
```bash
git clone https://github.com/Urabewe/DSRE---Digital-Sound-Resolution-Enhancer-English.git
cd DSRE---Digital-Sound-Resolution-Enhancer-English
```

### 2. Create Virtual Environment
```bash
python -m venv DSRE
```

### 3. Activate Virtual Environment

```bash
DSRE\Scripts\activate
```

### 4. Install Requirements
```bash
pip install -r requirements.txt
```

### 5. Run the Application
```bash
python dsre.py
```

## Notes
- Make sure your virtual environment is activated before installing requirements or running the application
- The application provides a GUI interface for batch audio enhancement
- Supports multiple audio formats: WAV, MP3, FLAC, M4A, etc.




---

## Parameters

| Parameter                               | Default | Description                                                   |
| -------------------------------------------- | ------------- | ------------------------------------------------------------------ |
| Modulation count (m)                  | 8             | Number of enhancement repetitions, higher = more detail |
| Decay amplitude                                 | 1.25          | High-frequency decay control                              |
| Pre-processing high-pass cutoff  | 3000 Hz       | Pre-enhancement high-pass filter                        |
| Post-processing high-pass cutoff | 16000 Hz      | Post-enhancement high-pass filter                       |
| Filter order                         | 11            | High-pass filter order                                   |
| Target sampling rate                 | 96000 Hz      | Output audio sample rate                                 |
| Output format                         | ALAC / FLAC / MP3 | Choose output format (Hi-Res or standard)       |

---
