import os
import platform
import datetime
import mimetypes
import hashlib
import re
import tkinter as tk
from tkinter import filedialog
import time
import sys
import threading 

OCR_TESSERACT_CONFIG = '--oem 1 --psm 3' 

try:
    from PIL import Image, ExifTags, UnidentifiedImageError
    try:
        import pytesseract
        if platform.system() == "Windows":
            tesseract_path_option1 = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
            tesseract_path_option2 = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe' 
            if os.path.exists(tesseract_path_option1):
                 pytesseract.pytesseract.tesseract_cmd = tesseract_path_option1
            elif os.path.exists(tesseract_path_option2):
                 pytesseract.pytesseract.tesseract_cmd = tesseract_path_option2

    except ImportError:
        pytesseract = None
        print("INFO: pytesseract not found. OCR for images will be disabled. Install Tesseract OCR engine and pytesseract python package (pip install pytesseract).")
except ImportError:
    Image = ExifTags = UnidentifiedImageError = None
    pytesseract = None 
    print("CRITICAL WARNING: Pillow (PIL) not found. Image metadata and text extraction will be disabled. Install with: pip install Pillow")

OPENCL_AVAILABLE = False
_np = None
try:
    import pyopencl as cl
    import numpy
    _np = numpy
    OPENCL_AVAILABLE = True
except ImportError:
    print("WARNING: PyOpenCL or NumPy not found. GPU acceleration for the demo task will be disabled. Install with: pip install pyopencl numpy")
except Exception as e:
    print(f"WARNING: PyOpenCL/NumPy found but could not initialize: {e}. GPU acceleration will be disabled.")

try:
    from geopy.geocoders import Nominatim
    from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
    GEOPY_AVAILABLE = True
except ImportError:
    GEOPY_AVAILABLE = False
    print("WARNING: geopy not found. GPS to Address conversion will be disabled. Install with: pip install geopy")

SUSPICIOUS_PATTERNS = [
    re.compile(r"<script.*?>.*?</script>", re.IGNORECASE | re.DOTALL),
    re.compile(r"eval\s*\(", re.IGNORECASE), re.compile(r"base64_decode\s*\(", re.IGNORECASE),
    re.compile(r"document\.write\s*\(", re.IGNORECASE), re.compile(r"powershell", re.IGNORECASE),
    re.compile(r"ActiveXObject", re.IGNORECASE), re.compile(r"vbscript", re.IGNORECASE),
    re.compile(r"function\s*\w+\s*\(.*?\)\s*\{.*?\}", re.IGNORECASE | re.DOTALL),
    re.compile(r"on[a-z]+\s*=", re.IGNORECASE), re.compile(r"<\?php", re.IGNORECASE),
]
SUSPICIOUS_KEYWORDS = [
    "malware", "virus", "trojan", "exploit", "shellcode", "payload",
    "obfuscate", "encrypt", "packed", "downloader", "dropper"
]

def _convert_to_degrees(value):
    d = float(value[0]); m = float(value[1]); s = float(value[2])
    return d + (m / 60.0) + (s / 3600.0)

def get_decimal_coords_from_exif_gps(gps_info):
    try:
        lat_ref = gps_info.get(1); lat_dms = gps_info.get(2)
        lon_ref = gps_info.get(3); lon_dms = gps_info.get(4)
        if lat_ref and lat_dms and lon_ref and lon_dms:
            decimal_lat = _convert_to_degrees(lat_dms)
            if lat_ref == 'S': decimal_lat = -decimal_lat
            decimal_lon = _convert_to_degrees(lon_dms)
            if lon_ref == 'W': decimal_lon = -decimal_lon
            return decimal_lat, decimal_lon
    except Exception as e: print(f"Error parsing GPSInfo: {e}")
    return None, None

def get_address_from_coords(lat, lon, user_agent="my_image_analyzer_app/1.1"):
    if not GEOPY_AVAILABLE: return " (geopy library not available)"
    if lat is None or lon is None: return " (Invalid coordinates)"
    try:
        geolocator = Nominatim(user_agent=user_agent)
        location = geolocator.reverse(f"{lat}, {lon}", exactly_one=True, timeout=10)
        return location.address if location else " (Address not found)"
    except GeocoderTimedOut: return " (Address lookup timed out)"
    except GeocoderUnavailable: return " (Address lookup service unavailable)"
    except Exception as e: return f" (Error during address lookup: {e})"

def get_opencl_simple_sum_gpu(data_bytes):
    if not OPENCL_AVAILABLE or not _np or not data_bytes:
        return None, "OpenCL or NumPy not available, or no data for GPU demo sum."
    try:
        platforms = cl.get_platforms()
        if not platforms: return None, "No OpenCL platforms found."
        devices = []
        for p in platforms: devices.extend(p.get_devices(cl.device_type.GPU))
        if not devices: devices = [d for p in platforms for d in p.get_devices(cl.device_type.ALL)] 
        if not devices: return None, "No OpenCL devices found."

        ctx = cl.Context([devices[0]])
        queue = cl.CommandQueue(ctx)
        mf = cl.mem_flags
        input_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=data_bytes)
        result_host_buffer = bytearray(8) 
        result_device_buf = cl.Buffer(ctx, mf.WRITE_ONLY, len(result_host_buffer))
        sum_kernel_src = """
        __kernel void simple_sum(__global const unsigned char *data,
                                 __global unsigned long *result, unsigned int N) {
            unsigned long sum = 0;
            for (unsigned int i = 0; i < N; ++i) { sum += data[i]; }
            *result = sum;
        }"""
        prg = cl.Program(ctx, sum_kernel_src).build()
        prg.simple_sum(queue, (1,), None, input_buf, result_device_buf, _np.uint32(len(data_bytes)))
        cl.enqueue_copy(queue, result_host_buffer, result_device_buf).wait()
        gpu_sum = int.from_bytes(result_host_buffer, byteorder=sys.byteorder)
        return f"GPU_SimpleByteSum:{gpu_sum}", f"OpenCL device for demo sum: {devices[0].name}"
    except Exception as e:
        return None, f"OpenCL Error (demo sum): {e}"

def get_basic_file_metadata(filepath):
    metadata = {}
    try:
        stat_info = os.stat(filepath)
        metadata["File Path"] = os.path.abspath(filepath)
        metadata["File Name"] = os.path.basename(filepath)
        metadata["File Size (bytes)"] = stat_info.st_size

        metadata["Modification Time"] = datetime.datetime.fromtimestamp(stat_info.st_mtime).strftime('%Y-%m-%d %H:%M:%S')

        mime_type, encoding = mimetypes.guess_type(filepath)
        metadata["MIME Type (guessed)"] = mime_type or "Unknown"

        try:
            import magic
            m = magic.Magic(mime=True, uncompress=True)
            metadata["MIME Type (libmagic)"] = m.from_file(filepath)
        except ImportError:
            metadata["MIME Type (libmagic)"] = "python-magic not installed"
        except Exception as e: 
             metadata["MIME Type (libmagic)"] = f"Error: {e}"

    except FileNotFoundError: metadata["Error"] = "File not found."
    except Exception as e: metadata["Error"] = f"An error occurred getting basic metadata: {e}"
    return metadata

def perform_file_hashing(filepath, results_dict):
    """Worker function for threaded hashing."""
    print("Hashing thread started...")
    hashes = {}
    gpu_sum_result = {"value": None, "info": "Skipped or error."}
    try:
        with open(filepath, 'rb') as f:
            file_content = f.read()

        if file_content:
            hashes["MD5 (CPU)"] = hashlib.md5(file_content).hexdigest()
            hashes["SHA1 (CPU)"] = hashlib.sha1(file_content).hexdigest()
            hashes["SHA256 (CPU)"] = hashlib.sha256(file_content).hexdigest()

            gpu_val, gpu_info = get_opencl_simple_sum_gpu(file_content)
            gpu_sum_result["value"] = gpu_val
            gpu_sum_result["info"] = gpu_info
        else:
            hashes["Error"] = "File is empty or could not be read for hashing."
            gpu_sum_result["info"] = "Skipped due to empty/unreadable file content."
    except Exception as e:
        hashes["Error"] = f"Error during hashing: {e}"
        gpu_sum_result["info"] = f"Skipped due to hashing error: {e}"

    results_dict['hashes'] = hashes
    results_dict['gpu_sum'] = gpu_sum_result
    print("Hashing thread finished.")

def analyze_content_for_suspicion(text_content):
    findings = []
    if not text_content or not isinstance(text_content, str): return findings
    for pattern in SUSPICIOUS_PATTERNS:
        try:
            if matches := pattern.findall(text_content): 
                for match in matches: findings.append(f"Suspicious pattern (regex: {pattern.pattern}): {str(match)[:100]}...")
        except Exception as e: findings.append(f"Error matching pattern {pattern.pattern}: {e}")
    for keyword in SUSPICIOUS_KEYWORDS:
        if re.search(r'\b' + re.escape(keyword) + r'\b', text_content, re.IGNORECASE):
            findings.append(f"Suspicious keyword: {keyword}")
    if len(text_content) > 100:
        from collections import Counter
        try:
            counts = Counter(text_content); text_len = len(text_content)
            log_func = _np.log2 if _np and hasattr(_np, 'log2') else (lambda x: __import__('math').log2(x))
            entropy = -sum((count / text_len) * log_func(count / text_len) for count in counts.values())
            if entropy > 4.8: findings.append(f"High entropy: {entropy:.2f} (potential obfuscation/encryption)")
        except Exception as e: findings.append(f"Entropy calculation error: {e}")
    return findings

def analyze_file(filepath):
    report_lines = []
    header = f"--- Analyzing Image File: {os.path.basename(filepath)} ---"
    print(f"\n{header}"); report_lines.append(header)

    metadata = get_basic_file_metadata(filepath)
    for key, value in metadata.items():
        line = f"{key}: {value}"; print(line); report_lines.append(line)
    if "Error" in metadata: return report_lines

    hashing_results = {}
    hashing_thread = threading.Thread(target=perform_file_hashing, args=(filepath, hashing_results))
    hashing_thread.start()
    print("Image processing (EXIF, OCR) starting on main thread...")

    extracted_image_info_parts = []
    ocr_text_content = ""
    current_type_info = "Type: Image File" 

    mime_type_primary = metadata.get("MIME Type (libmagic)", metadata.get("MIME Type (guessed)", "")).lower()
    file_ext = os.path.splitext(filepath)[1].lower()
    is_image = any(mime_type_primary.startswith(p) for p in ['image/']) or \
               any(file_ext == ext for ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.webp'])

    if not is_image:
        msg = f"\nSkipping detailed image analysis: File '{os.path.basename(filepath)}' not identified as a supported image type."
        msg += f"\n(MIME: {mime_type_primary}, Extension: {file_ext})"
        print(msg); report_lines.append(msg)
        hashing_thread.join() 
        if 'hashes' in hashing_results: 
            report_lines.append("\n--- Hashing & GPU Demo Sum ---")
            for k, v in hashing_results['hashes'].items(): report_lines.append(f"{k}: {v}")
            if hashing_results['gpu_sum']['value']: report_lines.append(f"GPU Demonstrative Value: {hashing_results['gpu_sum']['value']}")
            report_lines.append(f"GPU Process Info: {hashing_results['gpu_sum']['info']}")
        return report_lines

    if not Image or not ExifTags: 
        err_msg = "CRITICAL: Pillow library not available. Image processing disabled."
        print(err_msg); report_lines.append(err_msg)
        extracted_image_info_parts.append(err_msg)
    else:
        pil_image_obj = None
        try:
            with Image.open(filepath) as img:
                pil_image_obj = img.copy() 
                extracted_image_info_parts.append(f"Format: {img.format}, Mode: {img.mode}, Size: {img.size}")
                raw_exif = img._getexif()
                if raw_exif:
                    extracted_image_info_parts.append("EXIF Data:")
                    gpsinfo_dict = {}
                    for tag_id, value in raw_exif.items():
                        tag_name = ExifTags.TAGS.get(tag_id, tag_id)
                        if tag_name == "GPSInfo" and isinstance(value, dict):
                            for gps_tag_id, gps_val in value.items():
                                gps_tag_name_str = ExifTags.GPSTAGS.get(gps_tag_id, gps_tag_id)
                                gpsinfo_dict[gps_tag_id] = gps_val
                                extracted_image_info_parts.append(f"  GPS Tag {gps_tag_name_str} ({gps_tag_id}): {gps_val}")
                            if gpsinfo_dict:
                                lat, lon = get_decimal_coords_from_exif_gps(gpsinfo_dict)
                                if lat is not None and lon is not None:
                                    coords_str = f"  GPS Coords (Decimal): Lat {lat:.6f}, Lon {lon:.6f}"
                                    extracted_image_info_parts.append(coords_str); print(coords_str)
                                    addr = get_address_from_coords(lat, lon)
                                    addr_str = f"  Est. Address: {addr}"
                                    extracted_image_info_parts.append(addr_str); print(addr_str)
                                else: extracted_image_info_parts.append("  GPS Coords: Parse error")
                        else:
                            val_str = repr(value) if isinstance(value, bytes) else str(value)
                            extracted_image_info_parts.append(f"  {tag_name}: {val_str[:150]}") 
                else: extracted_image_info_parts.append("No EXIF data found.")

            if pytesseract and pil_image_obj:
                print("Attempting OCR on image (this may take a moment)...")
                try:
                    ocr_text_content = pytesseract.image_to_string(pil_image_obj, config=OCR_TESSERACT_CONFIG)
                    if ocr_text_content and ocr_text_content.strip():
                        extracted_image_info_parts.append(f"\nOCR Extracted Text:\n{ocr_text_content.strip()}")
                        print("OCR completed. Text found.")
                    else:
                        extracted_image_info_parts.append("No significant text found by OCR.")
                        print("OCR completed. No text found.")
                except pytesseract.TesseractNotFoundError:
                    err = "Tesseract OCR engine not found/configured. OCR disabled."
                    print(f"ERROR: {err}"); extracted_image_info_parts.append(err)
                except Exception as ocr_e:
                    err = f" (Error performing OCR: {ocr_e})"
                    print(err); extracted_image_info_parts.append(err)
            elif not pytesseract:
                 extracted_image_info_parts.append("OCR disabled: pytesseract library not available.")

        except UnidentifiedImageError if UnidentifiedImageError else Exception as img_err:
            err = f" (Pillow Error: Could not identify/open image: {img_err})"
            print(err); current_type_info += err; extracted_image_info_parts.append(err)
        except Exception as e:
            err = f" (Error processing image: {e})"
            print(err); current_type_info += err; extracted_image_info_parts.append(err)
        finally:
            if pil_image_obj: pil_image_obj.close()

    content_header = "\n--- Image Content Extraction & Analysis ---"
    print(content_header); report_lines.append(content_header)
    print(current_type_info); report_lines.append(current_type_info)

    full_extracted_text_for_report = "\n".join(extracted_image_info_parts)
    if full_extracted_text_for_report.strip():
        print("\nExtracted Image Info & OCR Snippet (first 1000 chars of combined data):")
        print(full_extracted_text_for_report[:1000])
        if len(full_extracted_text_for_report) > 1000: print("...\n(Content truncated)")
        report_lines.append("\n--- Full Extracted Image Info & OCR Text ---")
        report_lines.append(full_extracted_text_for_report)
    else:
        msg = "No specific image metadata or OCR text extracted."
        print(msg); report_lines.append(f"\n--- Full Extracted Image Info & OCR Text ---"); report_lines.append(msg)

    text_for_suspicion = full_extracted_text_for_report 

    print("Waiting for hashing thread to complete...")
    hashing_thread.join()
    report_lines.append("\n--- Hashing & GPU Demo Sum ---")
    if 'hashes' in hashing_results:
        for k, v in hashing_results['hashes'].items():
            line = f"{k}: {v}"; print(line); report_lines.append(line)
    if 'gpu_sum' in hashing_results:
        if hashing_results['gpu_sum']['value']:
            line = f"GPU Demonstrative Value: {hashing_results['gpu_sum']['value']}"; print(line); report_lines.append(line)
        line = f"GPU Process Info: {hashing_results['gpu_sum']['info']}"; print(line); report_lines.append(line)
    print("Main thread proceeding after hashing.")

    susp_header = "\n--- Suspicious Indicators (on Extracted Text) ---"
    print(susp_header); report_lines.append(susp_header)
    if text_for_suspicion.strip():
        findings = analyze_content_for_suspicion(text_for_suspicion)
        if findings:
            for finding in findings: print(f"- {finding}"); report_lines.append(f"- {finding}")
        else:
            msg = "No predefined suspicious indicators found in extracted text."; print(msg); report_lines.append(msg)
    else:
        msg = "No text available for rule-based suspicion analysis."; print(msg); report_lines.append(msg)

    report_lines.append("\n--- Analysis Complete ---")
    return report_lines

def main():
    root = tk.Tk(); root.withdraw()
    image_filetypes = [
        ("Image files", "*.jpg *.jpeg *.png *.gif *.bmp *.tiff *.tif *.webp"),
        ("All files", "*.*")
    ]
    filepath = filedialog.askopenfilename(title="Select an IMAGE file to analyze", filetypes=image_filetypes)
    if not filepath or not os.path.exists(filepath):
        print("No valid file selected or file not found. Exiting."); return

    start_time = time.perf_counter()
    report_data = analyze_file(filepath)
    duration = time.perf_counter() - start_time

    duration_str = f"Total Analysis Time: {duration:.4f}s"
    print(f"\n{duration_str}")
    report_data.insert(0, "="*50) 
    report_data.insert(0, duration_str) 

    base_fn = os.path.basename(filepath)
    report_fn_prefix = re.sub(r'[^\w-]', '_', base_fn)
    report_fn = f"Analysis_Report_{report_fn_prefix}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    final_report = [f"Analysis Report for Image: {base_fn}",
                    f"Full Path: {os.path.abspath(filepath)}",
                    f"Report Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"] + report_data
    try:
        with open(report_fn, "w", encoding="utf-8") as f: f.write("\n".join(final_report))
        print(f"Full report saved to: {os.path.abspath(report_fn)}")
    except Exception as e: print(f"Error saving report: {e}")

    print("\nNote: This script provides image info extraction & basic pattern matching.")
    print("GPU usage is for a DEMONSTRATIVE byte sum, NOT for OCR acceleration.")
    print("Interpret results with caution.")
    if not Image: print("CRITICAL: Pillow not found. Functionality severely limited.")
    if not pytesseract: print("INFO: Pytesseract/Tesseract OCR not found/configured. Text extraction from image content (OCR) was disabled.")

if __name__ == "__main__":

    if OPENCL_AVAILABLE and _np:
        print("--- OpenCL Initialization Info ---")
        try:

            plats = cl.get_platforms()
            print(f"Found {len(plats)} OpenCL platform(s). First is: {plats[0].name if plats else 'N/A'}")
        except Exception as e: print(f"Error during OpenCL enumeration: {e}")
        print("----------------------------------")

    main()