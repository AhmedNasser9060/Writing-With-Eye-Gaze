import os
import cv2
import numpy as np

# Try importing the necessary libraries for Arabic support
try:
    from PIL import Image, ImageFont, ImageDraw
    import arabic_reshaper
    from bidi.algorithm import get_display
    HAS_ARABIC_SUPPORT = True
except ImportError:
    HAS_ARABIC_SUPPORT = False

# Fallback font path for Windows (Tahoma is excellent for Arabic if available, Arial is another solid pick)
FONT_PATH = "C:\\Windows\\Fonts\\tahoma.ttf"
if not os.path.exists(FONT_PATH):
    FONT_PATH = "C:\\Windows\\Fonts\\arial.ttf"

_CURRENT_LANG = "en"

_TRANSLATIONS = {
    "Choose Language": "اختر اللغة",
    "Look Left for Arabic": "انظر يساراً للغة العربية",
    "Look Right for English": "انظر يميناً للغة الإنجليزية",
    "Eye Blink Keyboard": "لوحة مفاتيح العين",
    "To start the program": "لبدء البرنامج",
    "Close your eyes fully, then open.": "أغلق عينيك بالكامل، ثم افتحهما.",
    "Repeat 3 times (one full blink each).": "كرر 3 مرات (رمشة كاملة كل مرة).",
    "Left %d/%d": "يسار %d/%d",
    "Right %d/%d": "يمين %d/%d",
    
    # UI elements
    "Shortcuts": "اختصارات",
    "Keyboard": "لوحة مفاتيح",
    "Position: ": "الوضعية: ",
    
    # Shortcuts
    "Emergency": "طـوارئ",
    "Pain": "ألم",
    "Airway Obstruction": "انسداد التنفس",
    "Suction": "شفط",
    "Care": "رعاية",
    "Water": "ماء",
    "Food": "طعام",
    "Dizziness": "دوخة",
    "Can't Sleep": "أرق",
    "Change Position": "تغيير الوضعية",
    
    # Positions
    "Right": "اليمين",
    "Left": "اليسار",
    "Back": "الظهر",
    
    # Keyboard specials
    "SPACE": "مسافة",
    "ENTER": "إدخال",
    "DEL": "مسح",
    "BACK": "رجوع",
}

def set_language(lang: str):
    global _CURRENT_LANG
    _CURRENT_LANG = lang

def get_language() -> str:
    return _CURRENT_LANG

def tr(text: str) -> str:
    """Returns the translated strings based on current language; defaults to original text."""
    if _CURRENT_LANG == "ar":
        # Handle formatted strings translation by checking a prefix/suffix or exact match
        if text.startswith("Repeat ") and "times" in text:
            # specifically for the blinking setup text
            return _TRANSLATIONS.get("Repeat 3 times (one full blink each).", text)
        
        # Check standard dictionary
        return _TRANSLATIONS.get(text, text)
    return text

_FONT_CACHE = {}
def get_font(path, size):
    if (path, size) not in _FONT_CACHE:
        try:
            _FONT_CACHE[(path, size)] = ImageFont.truetype(path, size)
        except IOError:
            _FONT_CACHE[(path, size)] = ImageFont.load_default()
    return _FONT_CACHE[(path, size)]

_TEXT_CACHE = {}
def get_bidi_cached(text):
    if text not in _TEXT_CACHE:
        reshaped = arabic_reshaper.reshape(text)
        _TEXT_CACHE[text] = get_display(reshaped)
    return _TEXT_CACHE[text]

def put_text(img, text, org, scale, color, thickness=1, center=False, font_face=cv2.FONT_HERSHEY_SIMPLEX):
    """
    Unified text drawing wrapper. 
    If language is 'ar', it draws properly shaped text via PIL using an ROI for performance.
    """
    # If the user is missing the libraries or it's not Arabic text, we use fallback OpenCV
    if not HAS_ARABIC_SUPPORT or not any('\u0600' <= c <= '\u06FF' for c in str(text)):
        # For English or numbers fallback to cv2
        if center:
            (tw, th), _ = cv2.getTextSize(text, font_face, scale, thickness)
            org = (org[0] - tw // 2, org[1] + th // 2)
        cv2.putText(img, text, org, font_face, scale, color, thickness)
        return

    bidi_text = get_bidi_cached(text)
    pil_font_size = int(scale * 30) # approx conversion
    font = get_font(FONT_PATH, pil_font_size)

    # Calculate text layout for centering via getbbox (x0, y0, x1, y1)
    bbox = font.getbbox(bidi_text)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    if center:
        # Center the text exactly at 'org' x,y
        draw_x = org[0] - tw // 2
        draw_y = org[1] - th // 2 - (bbox[1] * 2)
    else:
        # For uncentered, adjust Y origin because OpenCV origin is BOTTOM-LEFT
        draw_x = org[0]
        draw_y = org[1] - th - bbox[1]

    # Extreme optimization: only process the ROI bounding box to avoid converting 1080p frames
    pad = 25
    y1 = max(0, int(draw_y - pad))
    y2 = min(img.shape[0], int(draw_y + th + bbox[1] + pad))
    x1 = max(0, int(draw_x - pad))
    x2 = min(img.shape[1], int(draw_x + tw + pad))

    if y2 <= y1 or x2 <= x1:
        return

    roi = img[y1:y2, x1:x2]
    is_gray = (len(roi.shape) == 2)
    
    if is_gray:
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)
    else:
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

    img_pil = Image.fromarray(roi_rgb)
    draw = ImageDraw.Draw(img_pil)

    # Normalize color to tuple (handling grayscale)
    if isinstance(color, tuple) or isinstance(color, list):
        color_rgb = (int(color[2]), int(color[1]), int(color[0])) if len(color) >= 3 else (int(color[0]), int(color[0]), int(color[0]))
    else:
        color_rgb = (int(color), int(color), int(color))

    draw.text((draw_x - x1, draw_y - y1), bidi_text, font=font, fill=color_rgb)

    # Modify the original OpenCV image buffer
    if is_gray:
        img_cv2 = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2GRAY)
    else:
        img_cv2 = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    np.copyto(roi, img_cv2)

def draw_rounded_rect(img, pt1, pt2, color, thickness=-1, r=15):
    """Draws a rounded rectangle using OpenCV lines and circles."""
    x1, y1 = pt1
    x2, y2 = pt2
    
    # Boundary checks for completely off-image rects
    if x2 <= x1 or y2 <= y1: return
    
    if thickness < 0:
        cv2.circle(img, (x1 + r, y1 + r), r, color, -1)
        cv2.circle(img, (x2 - r, y1 + r), r, color, -1)
        cv2.circle(img, (x1 + r, y2 - r), r, color, -1)
        cv2.circle(img, (x2 - r, y2 - r), r, color, -1)
        
        cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
    else:
        cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness)
        cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness)
        cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness)
        cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness)
        cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)
        cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
