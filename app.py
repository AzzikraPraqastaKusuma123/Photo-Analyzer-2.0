import os
import cv2
import numpy as np
import base64
import io
import requests
from flask import Flask, request, render_template, session, Response
from werkzeug.utils import secure_filename
from PIL import Image, ExifTags
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph
from fractions import Fraction
from urllib.parse import quote # <-- BARU: Diperlukan untuk URL encoding

# --- Import library analisis ---
try:
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
except (ImportError, FileNotFoundError):
    pytesseract = None
try:
    from deepface import DeepFace
except ImportError:
    DeepFace = None
try:
    import colorgram
except ImportError:
    colorgram = None
try:
    from pyzbar.pyzbar import decode as pyzbar_decode
except ImportError:
    pyzbar_decode = None
try:
    from langdetect import detect
    from textblob import TextBlob
except ImportError:
    detect = None
    TextBlob = None

# ==============================================================================
# KONFIGURASI APLIKASI
# ==============================================================================
app = Flask(__name__)
app.secret_key = 'ganti-dengan-kunci-rahasia-anda'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
MODEL_FOLDER = os.path.join(BASE_DIR, 'model')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

FACE_CASCADE_PATH = os.path.join(MODEL_FOLDER, 'haarcascade_frontalface_default.xml')
PROTOTXT_PATH = os.path.join(MODEL_FOLDER, 'MobileNetSSD_deploy.prototxt')
CAFFEMODEL_PATH = os.path.join(MODEL_FOLDER, 'MobileNetSSD_deploy.caffemodel')

try:
    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
except Exception as e:
    face_cascade = None
    print(f"Warning: Gagal memuat model deteksi wajah: {e}")

try:
    object_net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, CAFFEMODEL_PATH)
    CLASSES = ["background", "pesawat", "sepeda", "burung", "perahu", "botol", "bus", "mobil", "kucing", "kursi", "sapi", "meja makan", "anjing", "kuda", "motor", "orang", "tanaman pot", "domba", "sofa", "kereta", "monitor tv"]
except Exception as e:
    object_net = None
    print(f"Warning: Gagal memuat model deteksi objek: {e}")

# ==============================================================================
# FUNGSI-FUNGSI BANTUAN (TETAP SAMA)
# ==============================================================================
def cv2_to_base64(img):
    _, buffer = cv2.imencode('.jpg', img)
    return base64.b64encode(buffer).decode('utf-8')

def format_file_size(size_bytes):
    if size_bytes == 0: return "0 B"
    size_name = ("B", "KB", "MB", "GB")
    i = int(np.floor(np.log(size_bytes) / np.log(1024)))
    p = np.power(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_name[i]}"
    
def get_gps_info(exif_data):
    gps_info = {}
    if 34853 in exif_data:
        try:
            gps_ifd = exif_data.get_ifd(34853)
            lat_d, lon_d = gps_ifd.get(2), gps_ifd.get(4)
            lat_ref, lon_ref = gps_ifd.get(1), gps_ifd.get(3)
            if lat_d and lon_d and lat_ref and lon_ref:
                lat = (lat_d[0] + lat_d[1]/60 + lat_d[2]/3600) * (-1 if lat_ref == 'S' else 1)
                lon = (lon_d[0] + lon_d[1]/60 + lon_d[2]/3600) * (-1 if lon_ref == 'W' else 1)
                gps_info['link'] = f"https://www.google.com/maps?q={lat},{lon}"
                gps_info['koordinat'] = f"{lat:.6f}, {lon:.6f}"
                return gps_info
        except Exception: return None
    return None

def format_exif_data(exif_data):
    if not exif_data: return None
    TRANSLATIONS = {
        'Make': 'Merek Perangkat', 'Model': 'Model Perangkat', 'Software': 'Software/OS Version',
        'DateTimeOriginal': 'Waktu Pengambilan', 'ExposureTime': 'Shutter Speed',
        'FNumber': 'Bukaan Lensa (F-Stop)', 'ISOSpeedRatings': 'ISO', 'FocalLength': 'Focal Length',
        'FocalLengthIn35mmFilm': 'Focal Length (35mm)', 'LensModel': 'Model Lensa',
        'LensMake': 'Merek Lensa', 'Flash': 'Info Flash', 'ExifVersion': 'Versi EXIF',
        'PixelXDimension': 'Lebar Pixel', 'PixelYDimension': 'Tinggi Pixel'
    }
    formatted_data = []
    for tag_id, value in exif_data.items():
        tag_name = ExifTags.TAGS.get(tag_id, tag_id)
        if tag_name in ['GPSInfo', 'MakerNote']: continue
        label = TRANSLATIONS.get(tag_name, tag_name)
        if isinstance(value, bytes): value = value.decode(errors='ignore').strip('\x00')
        elif tag_name == 'ExposureTime' and isinstance(value, (float, int)): value = f"{Fraction(value).limit_denominator()} detik"
        elif tag_name == 'FNumber' and isinstance(value, (float, int)): value = f"f/{value}"
        formatted_data.append({'label': label, 'value': str(value)})
    return sorted(formatted_data, key=lambda x: x['label'])

# ==============================================================================
# FUNGSI ANALISIS UTAMA (DIMODIFIKASI)
# ==============================================================================
def analyze_image_file(image_path, original_filename, source_url=None):
    hasil = {'error': None}
    try:
        img_cv = cv2.imread(image_path)
        img_pil = Image.open(image_path)
        
        # Deteksi sumber gambar
        if 'whatsapp_image' in original_filename.lower():
            hasil['sumber_gambar'] = 'WhatsApp'
        else:
            hasil['sumber_gambar'] = 'Tidak Diketahui'
        
        # --- BARU: Menambahkan URL untuk pencarian terbalik ---
        if source_url:
            encoded_url = quote(source_url, safe='')
            hasil['reverse_search_url'] = f"https://lens.google.com/uploadbyurl?url={encoded_url}"
        else:
            hasil['reverse_search_url'] = None
            
        # Properti & Kualitas (Sama seperti sebelumnya)
        laplacian_var = cv2.Laplacian(cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
        blur_status = "Tajam" if laplacian_var > 100 else "Cukup Tajam" if laplacian_var > 50 else "Blur"
        hasil['properti'] = [
            {"label": "Ukuran File", "value": format_file_size(os.path.getsize(image_path)), "tooltip": "Ukuran file gambar di disk."},
            {"label": "Dimensi", "value": f"{img_pil.width} x {img_pil.height} px", "tooltip": "Lebar x Tinggi gambar dalam pixel."},
            {"label": "Tingkat Ketajaman", "value": f"{laplacian_var:.1f} ({blur_status})", "tooltip": "Skor ketajaman gambar. Semakin tinggi semakin tajam."},
        ]

        # Sisa analisis lainnya (warna, barcode, ocr, exif, wajah, objek) tetap sama...
        hasil['warna'] = [{'rgb': f"rgb{c.rgb}", 'proporsi': f"{c.proportion:.4f}"} for c in colorgram.extract(image_path, 6)] if colorgram else []
        hasil['barcode'] = [f"{b.type}: {b.data.decode('utf-8')}" for b in pyzbar_decode(img_pil)] or ["Tidak ada terdeteksi."] if pyzbar_decode else ["Pustaka pyzbar tidak terinstal."]
        if pytesseract:
            text = pytesseract.image_to_string(img_pil, lang='ind+eng').strip()
            hasil['teks_ocr'] = text if text else "Tidak ada teks yang dapat dideteksi."
            if text and detect and TextBlob:
                try:
                    lang = detect(text); sentiment = TextBlob(text).sentiment
                    hasil['teks_ocr_lang'] = lang.upper()
                    sentimen_val = "Positif" if sentiment.polarity > 0.1 else "Negatif" if sentiment.polarity < -0.1 else "Netral"
                    hasil['teks_ocr_sentimen'] = f"{sentimen_val} ({sentiment.polarity:.2f})"
                except Exception: hasil['teks_ocr_lang'], hasil['teks_ocr_sentimen'] = "N/A", "N/A"
        else: hasil['teks_ocr'] = "Pustaka Pytesseract tidak terkonfigurasi."

        exif_data = img_pil.getexif(); hasil['metadata'] = format_exif_data(exif_data)
        gps_data = get_gps_info(exif_data)
        if gps_data: hasil['properti'].append({"label": "Koordinat GPS", "value": gps_data['link'], "is_link": True})
        
        hasil['wajah'] = [];
        if face_cascade and DeepFace:
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY); faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)); face_count = 0
            for (x, y, w, h) in faces:
                face_count += 1; face_img = img_cv[y:y+h, x:x+w]
                try:
                    analysis = DeepFace.analyze(face_img, actions=['age', 'gender', 'emotion'], enforce_detection=False); info = analysis[0]
                    hasil['wajah'].append({'id': f"Wajah #{face_count}", 'emosi': info.get('dominant_emotion', 'N/A').capitalize(), 'gender': info.get('dominant_gender', 'N/A').capitalize(), 'usia': info.get('age', 'N/A')})
                    label = f"#{face_count}: {info.get('dominant_emotion', '')}"; cv2.rectangle(img_cv, (x, y), (x+w, y+h), (255, 200, 0), 2); cv2.putText(img_cv, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
                except Exception: cv2.rectangle(img_cv, (x, y), (x+w, y+h), (255, 200, 0), 2)
        
        hasil['objek'] = []
        if object_net:
            (h, w) = img_cv.shape[:2]; blob = cv2.dnn.blobFromImage(cv2.resize(img_cv, (300, 300)), 0.007843, (300, 300), 127.5); object_net.setInput(blob)
            detections = object_net.forward(); detected_objects = set()
            for i in np.arange(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5:
                    idx = int(detections[0, 0, i, 1]);
                    if idx < len(CLASSES):
                        detected_objects.add(CLASSES[idx]); box = detections[0, 0, i, 3:7] * np.array([w, h, w, h]); (startX, startY, endX, endY) = box.astype("int")
                        cv2.rectangle(img_cv, (startX, startY), (endX, endY), (0, 255, 0), 2)
            hasil['objek'] = sorted(list(detected_objects)) if detected_objects else ["Tidak ada terdeteksi."]

        ai_desc = f"Gambar ini dianalisis memiliki ketajaman {blur_status.lower()}. Terdeteksi {len(hasil['wajah'])} wajah dan {len([o for o in hasil['objek'] if o != 'Tidak ada terdeteksi.'])} jenis objek. Teks dan metadata juga telah diekstrak."
        if hasil['sumber_gambar'] == 'WhatsApp': ai_desc += " Karena gambar ini terdeteksi berasal dari WhatsApp, data metadata perangkat dan lokasi kemungkinan besar telah dihapus."
        hasil['deskripsi_ai'] = ai_desc
        hasil['gambar_visualisasi'] = cv2_to_base64(img_cv)

    except Exception as e:
        hasil['error'] = f"Terjadi kesalahan saat analisis: {str(e)}"
    return hasil

# ==============================================================================
# RUTE-RUTE FLASK (DIMODIFIKASI)
# ==============================================================================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    filepath, filename, error, source_url = None, "analysis", None, None
    original_filename = ""
    try:
        if 'file' in request.files and request.files['file'].filename != '':
            file = request.files['file']; original_filename = file.filename
            filename = secure_filename(original_filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename); file.save(filepath)
        elif 'url' in request.form and request.form['url'] != '':
            source_url = request.form['url'] # <-- BARU: Simpan URL sumber
            response = requests.get(source_url, stream=True, timeout=10); response.raise_for_status()
            original_filename = os.path.basename(source_url).split("?")[0] or "image_from_url.jpg"; filename = secure_filename(original_filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            with open(filepath, 'wb') as f: f.write(response.content)
        else:
            return render_template('index.html', error="Silakan unggah file atau masukkan URL.")
        
        # --- BARU: Mengirim source_url ke fungsi analisis ---
        hasil_analisis = analyze_image_file(filepath, original_filename, source_url=source_url) 
        
        if hasil_analisis.get('error'): return render_template('index.html', error=hasil_analisis['error'])

        session['analysis_results'] = hasil_analisis; session['filename'] = original_filename
        return render_template('index.html', hasil=hasil_analisis, filename=original_filename)
    except Exception as e:
        error = f"Gagal memproses permintaan: {str(e)}"
    finally:
        if filepath and os.path.exists(filepath):
            try: os.remove(filepath)
            except OSError as e: print(f"Error saat menghapus file: {e}")
    return render_template('index.html', error=error)

@app.route('/generate_pdf')
def generate_pdf():
    # Rute ini tidak perlu diubah
    hasil = session.get('analysis_results')
    filename = session.get('filename', 'report')
    if not hasil or hasil.get('error'): return "Tidak ada data untuk dibuat laporan.", 404
    buffer = io.BytesIO(); p = canvas.Canvas(buffer, pagesize=letter); width, height = letter
    p.setFont("Helvetica-Bold", 18); p.drawCentredString(width / 2.0, height - 50, f"Laporan Analisis Gambar")
    p.setFont("Helvetica", 12); p.drawCentredString(width / 2.0, height - 70, filename)
    img_data = base64.b64decode(hasil['gambar_visualisasi']); img_reader = ImageReader(io.BytesIO(img_data))
    img_w, img_h = img_reader.getSize(); aspect = img_h / float(img_w)
    display_w = 400; display_h = display_w * aspect
    p.drawImage(img_reader, (width - display_w) / 2.0, height - 100 - display_h, width=display_w, height=display_h, preserveAspectRatio=True)
    y_pos = height - 120 - display_h; styles = getSampleStyleSheet(); styleN = styles['Normal']; styleB = styles['h3']
    def draw_section(title, content_list):
        nonlocal y_pos; y_pos -= 25; p_title = Paragraph(title, styleB); w_title, h_title = p_title.wrapOn(p, width - 144, 100)
        p_title.drawOn(p, 72, y_pos); y_pos -= h_title
        for item in content_list:
            y_pos -= 15; p_content = Paragraph(item, styleN); w_content, h_content = p_content.wrapOn(p, width - 162, 100)
            if y_pos - h_content < 50: p.showPage(); p.setFont("Helvetica-Bold", 12); y_pos = height - 72
            p_content.drawOn(p, 82, y_pos); y_pos -= h_content
        return y_pos
    draw_section("Properti & Kualitas", [f"<b>{item['label']}:</b> {item['value']}" for item in hasil.get('properti', [])])
    if hasil.get('wajah'): draw_section("Analisis Wajah", [f"<b>{face['id']}:</b> Emosi: {face['emosi']}, Gender: {face['gender']}, Usia: ~{face['usia']} thn" for face in hasil.get('wajah', [])])
    if hasil.get('objek'): draw_section("Objek Terdeteksi", [", ".join(hasil.get('objek', []))])
    if hasil.get('metadata'): draw_section("Metadata Perangkat", [f"<b>{item['label']}:</b> {item['value']}" for item in hasil.get('metadata', [])])
    if hasil.get('teks_ocr') != "Tidak ada teks yang dapat dideteksi.": draw_section("Teks dari Gambar (OCR)", [hasil.get('teks_ocr', '')])
    p.save(); buffer.seek(0)
    return Response(buffer, mimetype='application/pdf', headers={'Content-Disposition': f'attachment;filename=Laporan_{filename}.pdf'})

# ==============================================================================
# MENJALANKAN APLIKASI
# ==============================================================================
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')