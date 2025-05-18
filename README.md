# Advanced Image Analyzer

This Python script analyzes image files, extracting a wealth of information including:

*   **Basic File Metadata:** Filename, path, size, timestamps, MIME type (guessed and via libmagic).
*   **Image-Specific Details (via Pillow):** Format, mode, dimensions.
*   **EXIF Data:** Extracts all available EXIF tags, with special handling for:
    *   **GPS Coordinates:** Converts GPS EXIF tags to decimal latitude/longitude.
    *   **Reverse Geocoding:** (Optional, via `geopy`) Converts GPS coordinates to a human-readable street address using Nominatim (OpenStreetMap).
*   **Optical Character Recognition (OCR) (via Tesseract & pytesseract):** Extracts text embedded within the image.
*   **File Hashes:** Calculates MD5, SHA1, and SHA256 hashes of the file content (CPU-based).
*   **GPU Demonstrative Calculation (Optional, via PyOpenCL & NumPy):** Performs a simple byte sum on the file content using an OpenCL-capable GPU (if available) to demonstrate GPU integration. *Note: This does NOT accelerate OCR.*
*   **Suspicious Content Detection:** Scans extracted text (EXIF and OCR) for predefined suspicious patterns and keywords, and checks for high entropy which might indicate obfuscation.
*   **Reporting:** Generates a comprehensive text report of the analysis and saves it to a file.

The script uses threading to perform file hashing concurrently with image processing (EXIF/OCR) to improve overall analysis time for a single image.

## Features

*   User-friendly Tkinter file dialog for selecting images.
*   Support for common image formats (JPG, PNG, GIF, BMP, TIFF, WebP).
*   Detailed EXIF parsing, including GPS to address conversion.
*   OCR for text extraction directly from image pixels.
*   CPU-based cryptographic hashing.
*   Optional GPU-accelerated demonstrative byte sum.
*   Rule-based suspicious pattern and keyword matching in extracted text.
*   Entropy calculation on extracted text.
*   Clear console output during analysis.
*   Saves a detailed analysis report to a timestamped text file.
*   Graceful handling of missing optional libraries with informative warnings.

## Prerequisites

### 1. Python 3
   Make sure you have Python 3.7+ installed.

### 2. Tesseract OCR Engine
   This script uses `pytesseract`, which is a Python wrapper for Google's Tesseract OCR Engine. **You must install Tesseract OCR separately.**

   *   **Windows:** Download the installer from the [Tesseract at UB Mannheim page](https://github.com/UB-Mannheim/tesseract/wiki). During installation, ensure you select to install language data (e.g., English) and add Tesseract to your system PATH.
   *   **Linux (Debian/Ubuntu):**
   * 
      ```bash
      sudo apt update
      sudo apt install tesseract-ocr tesseract-ocr-eng # For English
      ```
      (Replace `tesseract-ocr-eng` with other language packs as needed, e.g., `tesseract-ocr-deu` for German).
   *   **macOS:**
   * 
      ```bash
      brew install tesseract
      ```

### 3. libmagic (for `python-magic`)
   The `python-magic` library provides more accurate MIME type detection and relies on the `libmagic` C library.

   *   **Linux (Debian/Ubuntu):**
   * 
      ```bash
      sudo apt install libmagic1
      ```
   *   **macOS:**
   * 
      ```bash
      brew install libmagic
      ```
   *   **Windows:** The `python-magic-bin` package (installed via pip below) often bundles the necessary DLLs.

### 4. OpenCL (Optional, for GPU demonstrative sum)
   If you want to use the GPU demonstrative sum feature:
   *   Ensure you have up-to-date GPU drivers that support OpenCL.
   *   You might need to install an OpenCL SDK from your GPU vendor (NVIDIA CUDA Toolkit includes OpenCL, AMD ROCm or AMD APP SDK, Intel OpenCL SDK).

## Installation of Python Dependencies

It's highly recommended to use a Python virtual environment.

1.  **Create and activate a virtual environment (optional but recommended):**
    ```bash
    python3 -m venv image_analyzer_env
    # On Linux/macOS:
    source image_analyzer_env/bin/activate
    # On Windows (Command Prompt):
    # image_analyzer_env\Scripts\activate.bat
    # On Windows (PowerShell):
    # .\image_analyzer_env\Scripts\Activate.ps1
    ```

2.  **Install the required Python packages:**
    ```bash
    pip install Pillow pytesseract pyopencl numpy geopy python-magic python-magic-bin
    ```
    *   `Pillow`: Image processing.
    *   `pytesseract`: Python wrapper for Tesseract OCR.
    *   `pyopencl`: For OpenCL GPU interaction (optional).
    *   `numpy`: Required by PyOpenCL and for some calculations.
    *   `geopy`: For reverse geocoding (optional).
    *   `python-magic` / `python-magic-bin`: For MIME type detection (optional but recommended).

## Usage

1.  Save the script as `image_analyzer.py` (or any other `.py` name).
2.  Make it executable (on Linux/macOS):
    ```bash
    chmod +x image_analyzer.py
    ```
3.  Run the script from your terminal:
    ```bash
    python image_analyzer.py
    ```
    (Or `./image_analyzer.py` if it's executable and in your current directory/PATH on Linux/macOS).

4.  A file dialog will appear. Select the image file you want to analyze.
5.  The script will process the image and print analysis information to the console.
6.  A detailed report will be saved as a `.txt` file in the same directory where the script is located (e.g., `Analysis_Report_my_image.jpg_20231027_143000.txt`).

## Configuration

*   **Tesseract Path:** If `pytesseract` cannot find your Tesseract installation (especially on Windows if not added to PATH), you may need to manually set the path within the script. Find the line:
    ```python
    # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    ```
    Uncomment it and update the path to your `tesseract.exe`.

*   **OCR Configuration:** The `OCR_TESSERACT_CONFIG` variable at the top of the script can be modified to pass specific command-line options to Tesseract (e.g., for different languages or page segmentation modes). The default is `'--oem 1 --psm 3'` which generally provides good results with Tesseract 4+.

*   **User Agent for Geocoding:** The `get_address_from_coords` function uses a `user_agent` string. If you use this script frequently or in an automated way, consider changing it to something unique and descriptive that includes contact information, as per Nominatim's usage policy.

## Troubleshooting

*   **`TesseractNotFoundError`:**
    *   Ensure Tesseract OCR Engine is installed correctly.
    *   Ensure Tesseract is in your system's PATH, or set `pytesseract.pytesseract.tesseract_cmd` in the script.
*   **`pyopencl` issues:**
    *   Verify OpenCL drivers and SDK are installed.
    *   Installation of `pyopencl` can sometimes be complex. Refer to its documentation for platform-specific build instructions if `pip install` fails.
*   **`python-magic` errors (`failed to find libmagic`):**
    *   Ensure `libmagic` (or its equivalent) is installed on your system (see Prerequisites).
    *   On Windows, `python-magic-bin` usually works. If you installed `python-magic` and it fails, try `pip uninstall python-magic` then `pip install python-magic-bin`.
*   **Slow OCR:** OCR speed depends on image size, complexity, and Tesseract's performance. The `--oem 1` (LSTM engine) option in `OCR_TESSERACT_CONFIG` is generally faster and more accurate for Tesseract 4+ if the LSTM models are installed.
*   **Permissions:** Ensure the script has read permissions for the image file and write permissions in its own directory to save the report.

## Disclaimer

This script is for informational and educational purposes. While it attempts to identify suspicious patterns, it is **NOT** a comprehensive security or forensic tool. Interpret the results with caution and use professional tools for critical security analysis. The GPU functionality is a simple demonstration and does not significantly speed up the core image analysis (like OCR).

## License

This project is open source. Please feel free to use, modify, and distribute. (Consider adding a specific license like MIT or GPL if you wish).
