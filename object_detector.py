# -*- coding: utf-8 -*- # Bu setir türkmen harplaryny goldamak üçin möhüm bolup biler
import cv2
from ultralytics import YOLO
import time
from datetime import datetime
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont # Pillow modullaryny import etmek

# --- Sazlamalar ---
MODEL_NAME = 'yolov8n.pt' # YOLOv8 modeliniň ady (mysal üçin, yolov8n.pt, yolov8s.pt)
CONFIDENCE_THRESHOLD = 0.45 # Görkeziljek deteksiýa üçin minimal ynam baly (0.0-dan 1.0-a çenli)
WEBCAM_INDEX = 0 # Webkameranyň indeksi. Adatça esasy kamera üçin 0.
LOG_FILENAME = "detection_log_tk.txt" # Bellige alyş (log) faýlynyň ady

# --- Şrift Sazlamalary (MÖHÜM!) ---
# !!! Sistemäňizde bar bolan we türkmen harplaryny goldaýan şrift faýlynyň ýoluny saýlaň !!!
# Mysallar:
# FONT_PATH = "C:/Windows/Fonts/arial.ttf" # Windows Mysaly
# FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf" # Linux Mysaly
# FONT_PATH = "/System/Library/Fonts/Supplemental/Arial Unicode.ttf" # macOS Mysaly
FONT_PATH = "arial.ttf"  # Defolt - eger arial.ttf skript bilen bir papkada bolsa ýa-da doly ýoly görkeziň
FONT_SIZE = 18 # Şriftiň ölçegi, zerurlyga görä sazlaň
try:
    font_pil = ImageFont.truetype(FONT_PATH, FONT_SIZE)
    print(f"Şrift üstünlikli ýüklendi: {FONT_PATH}")
except IOError:
    print(f"!!!!!!!! ÝALŇYŞLYK: Şrift faýly şu ýerde tapylmady '{FONT_PATH}' !!!!!!!!")
    print("Haýyş edýäris, şrifti guruň ýa-da skriptdäki FONT_PATH üýtgediň.")
    print("Defolt Pillow şrifti ulanylýar (türkmen harplaryny goldamazlygy mümkin).")
    try:
        # Eger görkezilen şrift tapylmasa, defolt şrifti synap görüň
        font_pil = ImageFont.load_default()
    except Exception as e:
        print(f"Defolt şrifti ýükläp bolmady: {e}")
        font_pil = None # Eger defolt hem ýüklenmese, şrifti None ediň

# ---------------------

# --- Türkmençe Terjimeler ---
# !!! MÖHÜM: Bu sözlügiň takyk we doludygyna göz ýetiriň !!!
translations_tk = {
    "person": "adam", "bicycle": "tigir", "car": "maşyn", "motorcycle": "motosikl",
    "airplane": "samolýot", "bus": "awtobus", "train": "otly", "truck": "ýük maşyn",
    "boat": "gaýyk", "traffic light": "swetofor", "fire hydrant": "ýangyn gidranty",
    "stop sign": "durmak belgisi", "parking meter": "parkowka hasaplaýjy", "bench": "skameýka",
    "bird": "guş", "cat": "pişik", "dog": "it", "horse": "at", "sheep": "goýun",
    "cow": "sygyr", "elephant": "pil", "bear": "aýy", "zebra": "zebra", "giraffe": "žiraf",
    "backpack": "rukzak", "umbrella": "saýawan", "handbag": "el sumka", "tie": "galstuk",
    "suitcase": "çemodan", "frisbee": "frisbi", "skis": "lyža", "snowboard": "snoubord",
    "sports ball": "sport topy", "kite": "batbörek", "baseball bat": "beýsbol taýagy",
    "baseball glove": "beýsbol ellik", "skateboard": "skeýtbord", "surfboard": "sýorf doskasy",
    "tennis racket": "tennis raketkasy", "bottle": "çüýşe", "wine glass": "wino bulgury",
    "cup": "käse", "fork": "wilka", "knife": "pyçak", "spoon": "çemçe", "bowl": "jam",
    "banana": "banan", "apple": "alma", "sandwich": "buterbrod", "orange": "apelsin",
    "broccoli": "brokkoli", "carrot": "käşir", "hot dog": "hot dog", "pizza": "pitsa",
    "donut": "ponçik", "cake": "tort", "chair": "oturgyç", "couch": "diwan",
    "potted plant": "gülli düýp", "bed": "krowat", "dining table": "nahar stoly",
    "toilet": "hajathana", "tv": "telewizor", "laptop": "noutbuk", "mouse": "syçanjyk (kompýuter)",
    "remote": "pult", "keyboard": "klawiatura", "cell phone": "öýjükli telefon",
    "microwave": "mikrowolnowka", "oven": "peç", "toaster": "toster", "sink": "rakowina",
    "refrigerator": "holodilnik", "book": "kitap", "clock": "sagat", "vase": "waza",
    "scissors": "gaýçy", "teddy bear": "oýunjak aýy", "hair drier": "fen",
    "toothbrush": "diş çotgasy",
    "pen": "ruçka",
    "Unknown": "Näbelli"
}
# --------------------------

# --- Reňk Kesgitlemeleri (Pillow üçin RGB ulanyň) ---
pil_box_color = (0, 255, 0)   # Çarçuwa çyzygy üçin ýaşyl RGB
pil_text_color = (0, 0, 0)   # Tekst üçin gara RGB
pil_bg_color = (0, 255, 0)   # Tekst fony üçin ýaşyl RGB
box_thickness = 2            # Çarçuwa çyzygynyň galyňlygy
# ---------------------------------------------


print(f"Model ýüklenýär: {MODEL_NAME}...")
try:
    # YOLOv8 modelini ýüklemek
    model = YOLO(MODEL_NAME)
    # Modelden klas atlaryny almak (iňlisçe)
    class_names = model.names
    print("Model üstünlikli ýüklendi.")
except Exception as e:
    print(f"YOLO modelini ýüklemekde ýalňyşlyk: {e}")
    exit() # Ýalňyşlyk bolsa programmadan çykmak

print("Webkamera başlangyç ýagdaýa getirilýär...")
# Webkamera bilen baglanyşyk açmak
cap = cv2.VideoCapture(WEBCAM_INDEX)

# Kameranyň açylandygyny barlamak
if not cap.isOpened():
    print(f"Ýalňyşlyk: Webkamerany şu indeks bilen açyp bolmady {WEBCAM_INDEX}.")
    print("Eger birnäçe kameraňyz bar bolsa, WEBCAM_INDEX-i üýtgetmegi synap görüň.")
    exit()

# --- Loglamak (Bellige Almak) Sazlamalary ---
# Diňe täze peýda bolanlary bellige almak üçin *öňki* kadrdaky obýektleri yzarlaň
objects_in_previous_frame_tk = set()
print(f"Deteksiýalar bellige alynýar: {LOG_FILENAME}")
# Log faýlyny goşmaç ('a') režimde açyň, utf-8 kodirovkasy bilen
log_file = open(LOG_FILENAME, 'a', encoding='utf-8')
# Bu log sessiýasynyň başlan wagtyny görkezýän başlygy ýazyň
log_file.write(f"\n--- Log Sessiýasy Başlandy: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n")
log_file.flush() # Başlygyň derrew ýazylandygyna göz ýetiriň
# ---------------------

print("Real wagtda deteksiýa başlanýar... Çykmak üçin 'q' basyň.")

try: # Faýlyň ýalňyşlyk ýüze çyksa-da ýapylmagyny üpjün etmek üçin try...finally ulanyň
    while True:
        # Webkameradan bir kadr (frame) okamak
        success, frame_cv = cap.read() # Kadr OpenCV BGR formatynda alynýar
        # Kadryň üstünlikli alnandygyny barlamak
        if not success:
            print("Ýalňyşlyk: Webkameradan kadr alyp bolmady.")
            break # Eger kadr alynmassa, aýlawdan çykmak

        # --- Obýekt Deteksiýasyny Ýerine Ýetiriň ---
        # Kadry YOLO modeline bermek.
        # 'verbose=False' her kadr üçin ultralytics-den jikme-jik konsol çyktysyny öçürýär.
        # 'conf=CONFIDENCE_THRESHOLD' diňe ynam derejesi ýokary bolan netijeleri almak üçin.
        results = model(frame_cv, verbose=False, conf=CONFIDENCE_THRESHOLD)

        # --- Pillow bilen Çyzmaga Taýýarlamak ---
        # OpenCV BGR kadryny Pillow RGB kadryna öwürmek
        # OpenCV BGR reňk tertibini ulanýar, Pillow bolsa RGB.
        frame_pil = Image.fromarray(cv2.cvtColor(frame_cv, cv2.COLOR_BGR2RGB))
        # Pillow şekili üçin çyzgy kontekstini almak
        # Bu obýekt arkaly şekiliň üstüne çyzyp bolýar (çyzyklar, tekstler we ş.m.)
        draw = ImageDraw.Draw(frame_pil)

        # --- Deteksiýa Netijelerini Işläp Geçmek ---
        # Häzirki KADRDA ýüze çykarylan üýtgeşik türkmen atlaryny saklamak üçin set (toplum)
        detected_objects_tk_current = set()

        # Deteksiýa netijeleriniň üstünden geçmek
        for result in results:
            # Çarçuwa (bounding box) netijelerini almak
            boxes = result.boxes
            # Her bir tapylan çarçuwa üçin aýlanmak
            for box in boxes:
                # Çarçuwa Koordinatalaryny almak (çep, ýokarky, sag, aşaky)
                # .xyxy[0] birinji (we adatça ýeke-täk) şekil üçin koordinatalary berýär
                # map(int, ...) koordinatalary floatdan int-e öwürýär
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # Ynam Derejesini (Confidence) almak
                # .conf[0] bu deteksiýa üçin ynam derejesini float görnüşinde berýär
                confidence = float(box.conf[0])
                # Klas ID-sini we Adyny (Iňlisçe) almak
                # .cls[0] obýektiň klasynyň san belgisi (ID)
                cls_id = int(box.cls[0])
                # Modeliň klas atlaryndan (class_names) ID arkaly iňlisçe ady tapmak
                # .get() usuly, eger ID tapylmasa 'Unknown' gaýtarmak üçin howpsuzdyr
                english_name = class_names.get(cls_id, "Unknown")

                # --- Terjime ---
                # Iňlisçe ady türkmençe terjimeler sözlüginden (translations_tk) gözlemek
                # Eger terjime tapylmasa, asyl iňlisçe ady ulanmak
                turkmen_name = translations_tk.get(english_name, english_name)
                # Tapylan türkmen adyny häzirki kadryň toplumyna (set) goşmak
                detected_objects_tk_current.add(turkmen_name)

                # --- Çarçuwany Çyzmak (Pillow ulanyp) ---
                # draw.rectangle funksiýasy iki nokadyň koordinatasyny (ýokarky çep, aşaky sag) kabul edýär
                # 'outline' çarçuwanyň reňkini, 'width' çyzygyň galyňlygyny belleýär
                draw.rectangle( [(x1, y1), (x2, y2)], outline=pil_box_color, width=box_thickness)

                # --- Pillow bilen Belgini Taýýarlamak we Çyzmak ---
                if font_pil: # Diňe şrift üstünlikli ýüklenende teksti çyzyň
                    # Görkeziljek belginiň tekstini taýýarlamak (meselem: "maşyn: 0.85")
                    label = f"{turkmen_name}: {confidence:.2f}"

                    # Fon hasaplamasy üçin tekst ölçegini almak
                    # draw.textbbox başlangyç koordinata (meselem 0,0), tekst we şrift alýar
                    # we tekstiň daşyny gurşaýan gönüburçlugyň koordinatalaryny (çep, ýokarky, sag, aşaky) gaýtarýar
                    text_bbox = draw.textbbox((0, 0), label, font=font_pil) # Ölçeg üçin başlangyç nokady (0,0) möhüm däl
                    text_width = text_bbox[2] - text_bbox[0] # Tekstiň ini
                    text_height = text_bbox[3] - text_bbox[1] # Tekstiň beýikligi

                    # Fon gönüburçlugynyň ýerleşişini hasaplaň (çarçuwanyň ýokarsynda)
                    # Fony tekstden biraz giňräk we belenträk ediň
                    bg_x1 = x1 # Fonuň çep gyrasy çarçuwanyňky bilen deň
                    bg_y1 = y1 - text_height - (box_thickness + 2) # Çarçuwa çyzygynyň ýokarsynda + biraz boşluk (padding)
                    bg_x2 = x1 + text_width + 4 # Tekstiň saga boşlugy
                    bg_y2 = y1 - box_thickness # Çarçuwa çyzygynyň edil ýokarsynda

                    # Eger belgi ýokardan (ekranyň gyrasyndan) çyksa sazlaň
                    if bg_y1 < 0:
                        # Eger ýokardan çyksa, ýokarky gyradan aşakda ýerleşdiriň
                        bg_y2 = text_height + 3
                        bg_y1 = 0 # Ýokarky gyradan başlaň

                    # Pillow ulanyp fon gönüburçlugyny çyzyň
                    # 'fill' gönüburçlugyň içini doldurjak reňki belleýär
                    draw.rectangle([ (bg_x1, bg_y1) , (bg_x2, bg_y2) ], fill=pil_bg_color)

                    # Pillow ulanyp teksti çyzyň (başlangyç nokady tekstiň ýokarky çep burçy)
                    # Teksti fon gönüburçlugynyň içine bir az ýerleşdiriň (meselem, +2 piksel saga, +1 piksel aşak)
                    draw.text((bg_x1 + 2, bg_y1 + 1), label, font=font_pil, fill=pil_text_color)
                # else:
                    # Eger şrift ýüklenmese, belki ýönekeý çarçuwa belgisi çyzmaly? (Islege görä)
                    # print("Şrift ýüklenmedi, belgileri çyzyp bolmaýar.") # Ýa-da hiç zat etmezlik

        # --- Täze Peýda Bolan Obýektleri Bellige Almak ---
        # Häzirki kadrda bar bolan, ýöne öňki kadrda bolmadyk obýektleri tapyň
        # Toplum (set) aýyrmasy ulanylýar
        newly_detected_tk = detected_objects_tk_current - objects_in_previous_frame_tk

        # Eger täze ýüze çykarylan obýekt görnüşleri bar bolsa
        if newly_detected_tk:
            # Häzirki senäni we wagty almak we formatlamak
            timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # Täze tapylan her bir obýekt üçin
            # sorted() logdaky tertibiň yzygiderli bolmagyny üpjün edýär
            for new_obj_tk in sorted(list(newly_detected_tk)):
                # Log ýazgysyny taýýarlamak
                log_entry = f"{timestamp_str} - Peýda boldy: {new_obj_tk}\n" # "Appears:" -> "Peýda boldy:"
                # Ýazgyny faýla ýazmak
                log_file.write(log_entry)
            # Üýtgeşmeleri derrew diske ýazmak (buferleşdirmegiň öňüni almak)
            log_file.flush()

        # --- Indiki Kadr Üçin Ýagdaýy Täzelemek ---
        # Häzirki kadryň obýektleri indiki iterasiýa üçin öňki kadryň obýektleri bolýar
        objects_in_previous_frame_tk = detected_objects_tk_current

        # --- Pillow RGB kadryny (indi çarçuwalary WE teksti bar) Yzyna OpenCV BGR-e Öwürmek ---
        # Pillow şekilini (RGB) NumPy massiwine öwürmek, soňra reňk kanallaryny BGR-e öwürmek
        frame_display = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

        # --- Netijedäki Kadry Görkezmek ---
        # Işlenen kadry ekranda görkezmek
        cv2.imshow("Obyekt deteksiyasy | Atamyrat Shukurov", frame_display) # Pencere ady

        # --- Çykyş Şerti ---
        # 1 millisekund garaşmak we eger 'q' düwmesi basylan bolsa barlamak
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Çykylýar...") # "Exiting..." -> "Çykylýar..."
            break # Eger 'q' basylsa, esasy aýlawdan çykmak

finally: # Bu blok aýlaw adaty ýagdaýda ýa-da ýalňyşlyk sebäpli çyksa-da ýerine ýetirilýär
    # --- Arassalamak ---
    print("Resurslar ýapylýar...") # "Closing resources..." -> "Resurslar ýapylýar..."
    # Webkamera resursyny boşatmak
    if cap.isOpened():
        cap.release()
    # Ähli OpenCV penjirelerini ýapmak
    cv2.destroyAllWindows()
    # Eger log faýly açyk bolsa, oňa gutaryş ýazgysyny ýazmak we ýapmak
    if 'log_file' in locals() and log_file and not log_file.closed:
        log_file.write(f"--- Log Sessiýasy Tamamlandy: {datetime.now().strftime('%d.%m.%Y - %H:%M:%S')} ---\n")
        log_file.close() # Log faýlynyň dogry ýapylandygyna göz ýetiriň
    print(f"Log ýazdyryldy: {LOG_FILENAME}") # "Log saved to" -> "Log ýazdyryldy:"
    print("Programma ýapyldy.") # "Application closed." -> "Programma ýapyldy."