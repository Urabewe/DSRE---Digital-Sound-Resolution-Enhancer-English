# DSRE / Deep Sound Resolution Enhancer

## Description

DSRE is a **high-performance audio enhancement tool** that can batch-convert any audio files into **high-resolution (Hi-Res) audio**.
Inspired by Sony DSEE HX, it uses a **non-deep-learning frequency enhancement algorithm**, allowing fast processing of large batches without heavy computation.

**Key Features:**

* **Batch Processing**: Convert multiple audio files at once.
* **Multiple Formats**: Supports WAV, MP3, FLAC, M4A, etc.
* **Preserves Cover & Metadata**: No manual editing required.
* **Flexible Parameters**: Modulation count, decay, high-pass filter, etc.
* **Fast & Stable**: Does not rely on deep learning, fast processing.

---

## DSRE Installation Instructions

## Prerequisites
- Python 3.7 or higher
- Git

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
python main.py
```

## Notes
- Make sure your virtual environment is activated before installing requirements or running the application
- The application provides a GUI interface for batch audio enhancement
- Supports multiple audio formats: WAV, MP3, FLAC, M4A, etc.
- To deactivate the virtual environment when done, simply run: `deactivate`




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
| Output format                         | ALAC / FLAC   | Choose Hi-Res output format                       |

---
