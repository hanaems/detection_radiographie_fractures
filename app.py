"""
Application - Détection de Fractures
Modèles : EfficientNet B0 (classification) + YOLOv8 (détection)
"""
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse, RedirectResponse
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageDraw
from ultralytics import YOLO
import io, base64, uuid, uvicorn

# ── Cache images en mémoire (img_id -> bytes) ────────────────────────────────
image_cache: dict[str, bytes] = {}
MAX_CACHE = 20   # on garde les 20 dernières images

# ── EfficientNet ─────────────────────────────────────────────────────────────
class BoneFractureClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.efficientnet_b0(weights=None)
        self.features  = backbone.features
        self.avgpool   = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.4), nn.Linear(1280, 256),
            nn.ReLU(),       nn.Dropout(0.2),
            nn.Linear(256, 2),
        )
    def forward(self, x):
        x = self.features(x); x = self.avgpool(x)
        return self.classifier(torch.flatten(x, 1))

app = FastAPI()

print("📂 Chargement des modèles...")
eff_model = BoneFractureClassifier()
ck = torch.load("models/best_model.pth", map_location="cpu")
eff_model.load_state_dict(ck["model_state_dict"])
eff_model.eval()
print("✅ EfficientNet chargé!")

yolo_model = YOLO("models/best.pt")
print("✅ YOLOv8 chargé!")

tf = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

YOLO_COLORS = [
    (255,80,80),(80,200,120),(80,150,255),
    (255,180,50),(200,80,255),(50,220,220),(255,100,180),
]

# ── CSS commun ───────────────────────────────────────────────────────────────
CSS = """
* { margin:0; padding:0; box-sizing:border-box; }
body { font-family:'Inter',sans-serif; background:#0a0e27; color:#e4e8f0; min-height:100vh; }
.header { background:linear-gradient(135deg,#1e3a8a,#3b82f6); padding:18px 40px;
          box-shadow:0 4px 20px rgba(59,130,246,.3); }
.header-content { max-width:1400px; margin:0 auto; display:flex;
                  justify-content:space-between; align-items:center; }
.logo { display:flex; align-items:center; gap:12px; }
.logo-icon { width:40px; height:40px; background:white; border-radius:8px;
             display:flex; align-items:center; justify-content:center; font-size:24px; }
.logo-text h1 { font-size:20px; font-weight:600; color:white; }
.logo-text p  { font-size:12px; color:rgba(255,255,255,.8); }
.main { max-width:1400px; margin:0 auto; padding:28px 40px; }
.card { background:#151b3d; border-radius:12px; padding:24px;
        border:1px solid rgba(59,130,246,.2); }
.card-hd { display:flex; align-items:center; gap:12px; margin-bottom:18px;
           padding-bottom:14px; border-bottom:1px solid rgba(59,130,246,.2); }
.card-ic { width:36px; height:36px; background:linear-gradient(135deg,#3b82f6,#2563eb);
           border-radius:8px; display:flex; align-items:center; justify-content:center;
           font-size:20px; }
.card-ti { font-size:16px; font-weight:600; color:white; }
.grid2   { display:grid; grid-template-columns:1fr 1fr; gap:24px; }
.grid3   { display:grid; grid-template-columns:1fr 1fr 1fr; gap:24px; }

/* Sélecteur de modèle */
.switcher { display:flex; gap:10px; margin-bottom:24px; }
.sw-btn {
    flex:1; padding:12px 8px; border-radius:10px; cursor:pointer;
    border:2px solid rgba(59,130,246,.3); background:rgba(59,130,246,.05);
    color:#94a3b8; font-size:13px; font-weight:600; text-align:center;
    transition:all .25s; text-decoration:none; display:block;
}
.sw-btn:hover { border-color:#3b82f6; color:white; background:rgba(59,130,246,.15); }
.sw-btn.active { border-color:#3b82f6; background:rgba(59,130,246,.25); color:white; }
.sw-btn .sw-name { font-size:15px; margin-bottom:3px; }
.sw-btn .sw-desc { font-size:11px; font-weight:400; opacity:.7; }

/* Boutons */
.btn-primary { background:linear-gradient(135deg,#3b82f6,#2563eb); color:white;
               border:none; padding:14px 28px; border-radius:10px; cursor:pointer;
               font-size:15px; font-weight:600; width:100%; margin-top:20px;
               transition:all .25s; box-shadow:0 4px 16px rgba(59,130,246,.3);
               display:block; text-align:center; text-decoration:none; }
.btn-primary:hover { transform:translateY(-2px); box-shadow:0 6px 24px rgba(59,130,246,.4); }

/* Upload */
.upload-area { border:2px dashed rgba(59,130,246,.3); border-radius:12px;
               padding:50px 20px; text-align:center; cursor:pointer; transition:all .3s;
               background:rgba(59,130,246,.05); }
.upload-area:hover { border-color:#3b82f6; background:rgba(59,130,246,.1); }
input[type=file] { display:none; }
#preview { text-align:center; margin-top:16px; }
#preview img { max-width:100%; border-radius:10px; }

/* Barres de probabilité */
.bar-wrap { height:8px; background:rgba(59,130,246,.2); border-radius:4px; margin-top:6px; }
.bar-fill-red  { height:100%; border-radius:4px;
                 background:linear-gradient(90deg,#ef4444,#dc2626); }
.bar-fill-green{ height:100%; border-radius:4px;
                 background:linear-gradient(90deg,#10b981,#059669); }
.bar-fill-blue { height:100%; border-radius:4px;
                 background:linear-gradient(90deg,#3b82f6,#2563eb); }

/* Métriques */
.metric { background:rgba(59,130,246,.1); border:1px solid rgba(59,130,246,.3);
          border-radius:10px; padding:18px; margin-bottom:14px; }
.metric-lbl { font-size:12px; color:#94a3b8; text-transform:uppercase;
              letter-spacing:.05em; margin-bottom:6px; }
.metric-val { font-size:30px; font-weight:700; color:white; }
.metric-unit{ font-size:16px; color:#94a3b8; }

/* Banner résultat */
.banner { border-radius:12px; padding:22px 28px; margin-bottom:24px;
          display:flex; align-items:center; gap:20px;
          box-shadow:0 8px 32px rgba(0,0,0,.3); }
.banner-icon { font-size:44px; }
.banner h2   { font-size:26px; font-weight:700; color:white; margin-bottom:6px; }
.banner p    { font-size:14px; color:rgba(255,255,255,.85); }

/* Image box */
.img-box { background:#0a0e27; border-radius:12px; padding:16px; text-align:center; }
.img-box img { max-width:100%; border-radius:10px;
               box-shadow:0 8px 32px rgba(0,0,0,.5); }

/* Détection card */
.det-card { display:flex; justify-content:space-between; align-items:center;
            padding:12px 16px; border-radius:10px; margin-bottom:10px;
            border-width:1px; border-style:solid; }

/* Info items */
.info-row { display:flex; justify-content:space-between; padding:10px 0;
            border-bottom:1px solid rgba(59,130,246,.15); font-size:14px; }
.info-row:last-child { border:none; }
.info-lbl { color:#94a3b8; }
.info-val { color:white; font-weight:600; }

@media(max-width:1024px){ .grid2,.grid3{ grid-template-columns:1fr; } }
"""

HTML_HEAD = f"""<!DOCTYPE html><html>
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>AI Medical Imaging — Détection de Fractures</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>{CSS}</style>
</head>"""

def header(subtitle: str, badge: str = "") -> str:
    return f"""
<div class="header">
  <div class="header-content">
    <div class="logo">
      <div class="logo-icon">🦴</div>
      <div class="logo-text">
        <h1>AI Medical Imaging</h1>
        <p>{subtitle}</p>
      </div>
    </div>
    <div style="font-size:13px;color:rgba(255,255,255,.8);">{badge}</div>
  </div>
</div>"""

def switcher_html(img_id: str, active: str) -> str:
    """Barre de changement de modèle affichée sur toutes les pages résultat."""
    modes = [
        ("efficientnet", "🧠 EfficientNet", "Classification"),
        ("yolo",         "🎯 YOLOv8",       "Détection"),
        ("combined",     "🔗 Les deux",      "Combiné"),
    ]
    btns = ""
    for key, name, desc in modes:
        cls = "sw-btn active" if key == active else "sw-btn"
        btns += (f'<a class="{cls}" href="/switch?img={img_id}&model={key}">'
                 f'<div class="sw-name">{name}</div>'
                 f'<div class="sw-desc">{desc}</div></a>')
    return f"""
<div class="card" style="margin-bottom:20px;">
  <div class="card-hd">
    <div class="card-ic">🔄</div>
    <div class="card-ti">Changer de modèle — même image</div>
  </div>
  <div class="switcher">{btns}</div>
  <div style="font-size:12px;color:#64748b;">
    L'image est conservée en mémoire. Aucun re-upload nécessaire.
  </div>
</div>"""


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE D'ACCUEIL
# ═══════════════════════════════════════════════════════════════════════════════
@app.get("/", response_class=HTMLResponse)
def home():
    return HTML_HEAD + """
<body>""" + header("Système de Détection Automatique de Fractures", "") + """
<div class="main">
  <!-- Info barre -->
  <div class="card" style="margin-bottom:20px;padding:16px 24px;">
    <div style="display:flex;gap:40px;flex-wrap:wrap;font-size:14px;">
      <div><div style="color:#94a3b8;font-size:11px;text-transform:uppercase;">Type d'examen</div>
           <div style="font-weight:600;">Radiographie osseuse</div></div>
      <div><div style="color:#94a3b8;font-size:11px;text-transform:uppercase;">Type d'analyse</div>
           <div style="font-weight:600;">Détection de fracture par IA</div></div>
      <div><div style="color:#94a3b8;font-size:11px;text-transform:uppercase;">Segments couverts</div>
           <div style="font-weight:600;">Coude · Doigts · Avant-bras · Humérus · Épaule · Poignet</div></div>
    </div>
  </div>

  <form method="post" action="/predict" enctype="multipart/form-data">
    <input type="hidden" name="selected_model" id="selected_model" value="efficientnet">
    <div class="grid2">

      <!-- Colonne gauche -->
      <div class="card">
        <div class="card-hd"><div class="card-ic">🤖</div>
          <div class="card-ti">Choisir le modèle d'analyse</div></div>

        <div class="switcher" style="margin-bottom:24px;">
          <div class="sw-btn active" id="btn-efficientnet" onclick="sel('efficientnet')">
            <div class="sw-name">🧠 EfficientNet B0</div>
            <div class="sw-desc">Classification · Fracture / Normal</div>
          </div>
          <div class="sw-btn" id="btn-yolo" onclick="sel('yolo')">
            <div class="sw-name">🎯 YOLOv8</div>
            <div class="sw-desc">Détection · Localise la fracture</div>
          </div>
          <div class="sw-btn" id="btn-combined" onclick="sel('combined')">
            <div class="sw-name">🔗 Les deux</div>
            <div class="sw-desc">Mode combiné</div>
          </div>
        </div>

        <div class="card-hd"><div class="card-ic">📸</div>
          <div class="card-ti">Image radiographique</div></div>

        <div class="upload-area" onclick="document.getElementById('fi').click()">
          <div style="font-size:44px;margin-bottom:12px;">⬆️</div>
          <div style="font-size:15px;color:#94a3b8;margin-bottom:6px;">
            <strong style="color:#3b82f6;">Cliquez</strong> ou glissez une radiographie</div>
          <div style="font-size:12px;color:#64748b;">JPG · PNG · JPEG</div>
        </div>
        <input type="file" name="file" id="fi" accept="image/*" required onchange="prev()">
        <div id="preview"></div>
        <button type="submit" class="btn-primary">🔍 LANCER L'ANALYSE IA</button>
      </div>

      <!-- Colonne droite -->
      <div class="card">
        <div class="card-hd"><div class="card-ic">📋</div>
          <div class="card-ti">Guide d'utilisation</div></div>
        <div style="display:flex;flex-direction:column;gap:14px;">

          <div style="display:flex;gap:14px;align-items:flex-start;">
            <div style="font-size:28px;min-width:36px;">🧠</div>
            <div>
              <div style="font-weight:600;color:white;margin-bottom:3px;">Analyse globale</div>
              <div style="font-size:13px;color:#94a3b8;line-height:1.6;">
                Détecte la présence ou l'absence d'une fracture sur toute la radiographie.
                Résultat immédiat avec niveau de certitude.</div>
            </div>
          </div>

          <div style="display:flex;gap:14px;align-items:flex-start;">
            <div style="font-size:28px;min-width:36px;">🎯</div>
            <div>
              <div style="font-weight:600;color:white;margin-bottom:3px;">Localisation</div>
              <div style="font-size:13px;color:#94a3b8;line-height:1.6;">
                Identifie et encadre les zones suspectes sur l'image.
                Précise le segment osseux concerné (coude, poignet, épaule…).</div>
            </div>
          </div>

          <div style="display:flex;gap:14px;align-items:flex-start;">
            <div style="font-size:28px;min-width:36px;">🔗</div>
            <div>
              <div style="font-weight:600;color:white;margin-bottom:3px;">Double vérification</div>
              <div style="font-size:13px;color:#94a3b8;line-height:1.6;">
                Les deux analyses sont exécutées simultanément.
                En cas de résultats contradictoires, une alerte est affichée.</div>
            </div>
          </div>

          <div style="margin-top:6px;padding:14px 16px;
                      background:rgba(59,130,246,.08);
                      border-left:3px solid #3b82f6;border-radius:0 8px 8px 0;
                      font-size:12px;color:#94a3b8;line-height:1.7;">
            ⚕️ Cet outil est une aide au diagnostic.
            Il ne remplace pas l'avis d'un médecin radiologue.
          </div>
        </div>
      </div>
    </div>
  </form>
</div>

<script>
function sel(m){
  document.getElementById('selected_model').value = m;
  ['efficientnet','yolo','combined'].forEach(k=>{
    document.getElementById('btn-'+k).classList.toggle('active', k===m);
  });
}
function prev(){
  const f = document.getElementById('fi').files[0];
  if(f){ const r=new FileReader();
    r.onload=e=>{ document.getElementById('preview').innerHTML=
      '<img src="'+e.target.result+'" style="margin-top:14px;">'; };
    r.readAsDataURL(f); }
}
</script>
</body></html>"""


# ═══════════════════════════════════════════════════════════════════════════════
# UPLOAD + DISPATCH
# ═══════════════════════════════════════════════════════════════════════════════
@app.post("/predict", response_class=HTMLResponse)
async def predict(file: UploadFile = File(...),
                  selected_model: str = Form("efficientnet")):
    contents = await file.read()
    # Stocker en cache
    img_id = str(uuid.uuid4())
    if len(image_cache) >= MAX_CACHE:
        oldest = next(iter(image_cache))
        del image_cache[oldest]
    image_cache[img_id] = contents

    image = Image.open(io.BytesIO(contents)).convert("RGB")
    return render(image, contents, img_id, selected_model)


@app.get("/switch", response_class=HTMLResponse)
def switch(img: str, model: str):
    """Changer de modèle sans re-uploader."""
    if img not in image_cache:
        return RedirectResponse("/")
    contents = image_cache[img]
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    return render(image, contents, img, model)


# ═══════════════════════════════════════════════════════════════════════════════
# MOTEUR DE RENDU
# ═══════════════════════════════════════════════════════════════════════════════
def render(image: Image.Image, contents: bytes, img_id: str, mode: str) -> str:
    if mode == "yolo":
        return render_yolo(image, contents, img_id)
    elif mode == "combined":
        return render_combined(image, contents, img_id)
    else:
        return render_efficientnet(image, contents, img_id)


# ─── EfficientNet ─────────────────────────────────────────────────────────────
def run_efficientnet(image: Image.Image):
    img_t = tf(image).unsqueeze(0)
    with torch.no_grad():
        out   = eff_model(img_t)
        probs = torch.nn.functional.softmax(out, dim=1).squeeze()
    pred_idx  = int(torch.argmax(probs))
    fractured = pred_idx == 0
    conf      = float(probs[pred_idx]) * 100
    p_frac    = float(probs[0]) * 100
    p_norm    = float(probs[1]) * 100
    return fractured, conf, p_frac, p_norm


def render_efficientnet(image, contents, img_id):
    fractured, conf, p_frac, p_norm = run_efficientnet(image)
    img_b64  = base64.b64encode(contents).decode()
    bg       = "linear-gradient(135deg,#dc2626,#991b1b)" if fractured else "linear-gradient(135deg,#10b981,#059669)"
    icon     = "🚨" if fractured else "✅"
    title    = "FRACTURE DÉTECTÉE" if fractured else "AUCUNE FRACTURE DÉTECTÉE"
    pred_lbl = "Fracture" if fractured else "Normal"

    return HTML_HEAD + f"""
<body>
{header("Résultat de l'analyse", f"Confiance : {conf:.1f} %")}
<div class="main">
  {switcher_html(img_id, "efficientnet")}
  <div class="banner" style="background:{bg};">
    <div class="banner-icon">{icon}</div>
    <div><h2>{title}</h2>
      <p>Analyse globale de la radiographie</p></div>
  </div>
  <div class="grid2">
    <div class="card">
      <div class="card-hd"><div class="card-ic">📸</div>
        <div class="card-ti">Radiographie analysée</div></div>
      <div class="img-box"><img src="data:image/jpeg;base64,{img_b64}"></div>
    </div>
    <div class="card">
      <div class="card-hd"><div class="card-ic">📊</div>
        <div class="card-ti">Métriques EfficientNet</div></div>
      <div class="metric">
        <div class="metric-lbl">Prédiction</div>
        <div class="metric-val">{pred_lbl}</div>
      </div>
      <div class="metric">
        <div class="metric-lbl">Confiance</div>
        <div class="metric-val">{conf:.1f}<span class="metric-unit"> %</span></div>
      </div>
      <div style="margin-top:8px;">
        <div style="display:flex;justify-content:space-between;font-size:13px;margin-bottom:4px;">
          <span style="color:#94a3b8;">🦴 Probabilité Fracture</span>
          <span style="color:white;font-weight:700;">{p_frac:.1f} %</span>
        </div>
        <div class="bar-wrap"><div class="bar-fill-red" style="width:{p_frac}%;"></div></div>
        <div style="display:flex;justify-content:space-between;font-size:13px;
                    margin-top:12px;margin-bottom:4px;">
          <span style="color:#94a3b8;">✅ Probabilité Normal</span>
          <span style="color:white;font-weight:700;">{p_norm:.1f} %</span>
        </div>
        <div class="bar-wrap"><div class="bar-fill-green" style="width:{p_norm}%;"></div></div>
      </div>
      <a href="/" class="btn-primary" style="margin-top:24px;">🔄 Nouvelle analyse</a>
    </div>
  </div>
</div>
</body></html>"""


# ─── YOLOv8 ───────────────────────────────────────────────────────────────────
def run_yolo(image: Image.Image):
    results = yolo_model(image, verbose=False)[0]
    boxes   = results.boxes
    img_draw = image.copy()
    draw     = ImageDraw.Draw(img_draw)
    detections = []
    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0]) * 100
            label  = yolo_model.names[cls_id]
            x1,y1,x2,y2 = [int(v) for v in box.xyxy[0]]
            color  = YOLO_COLORS[cls_id % len(YOLO_COLORS)]
            draw.rectangle([x1,y1,x2,y2], outline=color, width=3)
            text = f"{label} {conf:.0f}%"
            draw.rectangle([x1, y1-22, x1+len(text)*8+6, y1], fill=color)
            draw.text((x1+3, y1-20), text, fill="white")
            detections.append({"label":label, "conf":conf, "color":color})
    buf = io.BytesIO()
    img_draw.save(buf, format="JPEG", quality=90)
    img_b64 = base64.b64encode(buf.getvalue()).decode()
    has_fracture = any(
        "fracture" in d["label"].lower() or "positive" in d["label"].lower()
        for d in detections
    )
    return detections, has_fracture, img_b64


def render_yolo(image, contents, img_id):
    detections, has_fracture, img_b64 = run_yolo(image)
    nb   = len(detections)
    bg   = "linear-gradient(135deg,#dc2626,#991b1b)" if has_fracture else "linear-gradient(135deg,#10b981,#059669)"
    icon = "🚨" if has_fracture else "✅"
    title = f"{nb} ANOMALIE(S) DÉTECTÉE(S)" if nb > 0 else "AUCUNE ANOMALIE DÉTECTÉE"

    det_html = ""
    if detections:
        for d in detections:
            r,g,b = d["color"]
            det_html += f"""
<div class="det-card"
     style="background:rgba({r},{g},{b},.1);border-color:rgba({r},{g},{b},.4);">
  <div style="font-weight:600;color:white;">🦴 {d['label']}</div>
  <div style="font-size:22px;font-weight:700;color:rgb({r},{g},{b});">{d['conf']:.0f}%</div>
</div>"""
    else:
        det_html = '<div style="text-align:center;color:#94a3b8;padding:28px;">Aucune détection</div>'

    return HTML_HEAD + f"""
<body>
{header("Résultat de l'analyse", f"{nb} zone(s) détectée(s)")}
<div class="main">
  {switcher_html(img_id, "yolo")}
  <div class="banner" style="background:{bg};">
    <div class="banner-icon">{icon}</div>
    <div><h2>{title}</h2>
      <p>Localisation des zones suspectes sur la radiographie</p></div>
  </div>
  <div class="grid2">
    <div class="card">
      <div class="card-hd"><div class="card-ic">📸</div>
        <div class="card-ti">Radiographie annotée</div></div>
      <div class="img-box"><img src="data:image/jpeg;base64,{img_b64}"></div>
    </div>
    <div class="card">
      <div class="card-hd"><div class="card-ic">📊</div>
        <div class="card-ti">Zones détectées</div></div>
      {det_html}
      <a href="/" class="btn-primary" style="margin-top:16px;">🔄 Nouvelle analyse</a>
    </div>
  </div>
</div>
</body></html>"""


# ─── Combiné ──────────────────────────────────────────────────────────────────
def render_combined(image, contents, img_id):
    # Lancer les deux modèles
    fractured_eff, conf_eff, p_frac, p_norm = run_efficientnet(image)
    detections, has_fracture_yolo, img_yolo_b64 = run_yolo(image)
    img_orig_b64 = base64.b64encode(contents).decode()

    # Accord ou désaccord ?
    agree = (fractured_eff == has_fracture_yolo)

    # ── Désaccord ────────────────────────────────────────────────────────────
    if not agree:
        eff_verdict = "Fracture" if fractured_eff else "Normal"
        yolo_verdict = f"{len(detections)} détection(s)" if has_fracture_yolo else "Aucune détection"
        return HTML_HEAD + f"""
<body>
{header("Résultat de l'analyse — Perturbation", "⚠️ Image non concluante")}
<div class="main">
  {switcher_html(img_id, "combined")}

  <!-- Bannière alerte -->
  <div class="banner" style="background:linear-gradient(135deg,#b45309,#92400e);">
    <div class="banner-icon">⚠️</div>
    <div>
      <h2>PERTURBATION D'ANALYSE</h2>
      <p>L'analyse combinée ne peut pas conclure sur cette image</p>
    </div>
  </div>

  <!-- Explication -->
  <div class="card" style="margin-bottom:24px;">
    <div class="card-hd"><div class="card-ic">🔍</div>
      <div class="card-ti">Détail du désaccord</div></div>
    <div class="grid2">
      <div>
        <div style="font-size:12px;color:#94a3b8;text-transform:uppercase;margin-bottom:8px;">
          🧠 EfficientNet B0</div>
        <div style="font-size:22px;font-weight:700;
                    color:{'#ef4444' if fractured_eff else '#10b981'};">
          {eff_verdict}</div>
        <div style="font-size:14px;color:#94a3b8;margin-top:4px;">
          Confiance : {conf_eff:.1f} %</div>
        <div class="bar-wrap" style="margin-top:8px;">
          <div class="{'bar-fill-red' if fractured_eff else 'bar-fill-green'}"
               style="width:{conf_eff}%;"></div>
        </div>
      </div>
      <div>
        <div style="font-size:12px;color:#94a3b8;text-transform:uppercase;margin-bottom:8px;">
          🎯 YOLOv8</div>
        <div style="font-size:22px;font-weight:700;
                    color:{'#ef4444' if has_fracture_yolo else '#10b981'};">
          {yolo_verdict}</div>
        <div style="font-size:14px;color:#94a3b8;margin-top:4px;">
          {'Anomalie(s) localisée(s)' if has_fracture_yolo else 'Aucune zone suspecte'}</div>
      </div>
    </div>
    <div style="margin-top:20px;padding:14px;background:rgba(245,158,11,.08);
                border:1px solid rgba(245,158,11,.35);border-radius:8px;
                font-size:13px;color:#fbbf24;">
      💡 <strong>Recommandation :</strong> Vérifier la qualité et l'orientation de la radiographie,
      ou soumettre une image de meilleure résolution. En cas de doute clinique, consulter un radiologue.
    </div>
  </div>

  <div class="grid2">
    <div class="card">
      <div class="card-hd"><div class="card-ic">📸</div>
        <div class="card-ti">Image originale</div></div>
      <div class="img-box"><img src="data:image/jpeg;base64,{img_orig_b64}"></div>
    </div>
    <div class="card">
      <div class="card-hd"><div class="card-ic">🎯</div>
        <div class="card-ti">Image annotée YOLO</div></div>
      <div class="img-box"><img src="data:image/jpeg;base64,{img_yolo_b64}"></div>
    </div>
  </div>
  <a href="/" class="btn-primary" style="margin-top:24px;max-width:300px;margin-left:auto;
     margin-right:auto;display:block;">🔄 Nouvelle analyse</a>
</div>
</body></html>"""

    # ── Accord ───────────────────────────────────────────────────────────────
    bg    = "linear-gradient(135deg,#dc2626,#991b1b)" if fractured_eff else "linear-gradient(135deg,#10b981,#059669)"
    icon  = "🚨" if fractured_eff else "✅"
    title = "FRACTURE CONFIRMÉE PAR LES DEUX MODÈLES" if fractured_eff else "ABSENCE DE FRACTURE CONFIRMÉE"

    det_html = ""
    for d in detections:
        r,g,b = d["color"]
        det_html += f"""
<div class="det-card"
     style="background:rgba({r},{g},{b},.1);border-color:rgba({r},{g},{b},.4);">
  <span style="font-weight:600;color:white;">🦴 {d['label']}</span>
  <span style="font-size:20px;font-weight:700;color:rgb({r},{g},{b});">{d['conf']:.0f}%</span>
</div>"""
    if not det_html:
        det_html = '<div style="color:#94a3b8;text-align:center;padding:20px;">Aucune zone localisée par YOLO</div>'

    return HTML_HEAD + f"""
<body>
{header("Résultat de l'analyse", f"Confiance : {conf_eff:.1f}% · {len(detections)} zone(s) localisée(s)")}
<div class="main">
  {switcher_html(img_id, "combined")}

  <div class="banner" style="background:{bg};">
    <div class="banner-icon">{icon}</div>
    <div>
      <h2>{title}</h2>
      <p>Les deux analyses confirment le même résultat</p>
    </div>
  </div>

  <!-- Badge accord -->
  <div style="text-align:center;margin-bottom:20px;">
    <span style="display:inline-block;background:rgba(16,185,129,.15);
                 border:1px solid rgba(16,185,129,.4);color:#10b981;
                 padding:8px 20px;border-radius:20px;font-size:13px;font-weight:600;">
      ✓ Les deux analyses sont en accord — résultat fiable
    </span>
  </div>

  <div class="grid3">
    <!-- Image originale -->
    <div class="card">
      <div class="card-hd"><div class="card-ic">📸</div>
        <div class="card-ti">Image originale</div></div>
      <div class="img-box"><img src="data:image/jpeg;base64,{img_orig_b64}"></div>
    </div>

    <!-- Image YOLO -->
    <div class="card">
      <div class="card-hd"><div class="card-ic">🎯</div>
        <div class="card-ti">Localisation YOLO</div></div>
      <div class="img-box"><img src="data:image/jpeg;base64,{img_yolo_b64}"></div>
    </div>

    <!-- Métriques -->
    <div class="card">
      <div class="card-hd"><div class="card-ic">📊</div>
        <div class="card-ti">Résumé</div></div>

      <div class="metric" style="margin-bottom:10px;">
        <div class="metric-lbl">Verdict</div>
        <div class="metric-val" style="font-size:22px;">
          {'Fracture' if fractured_eff else 'Normal'}</div>
      </div>
      <div style="font-size:13px;color:#94a3b8;margin-bottom:4px;">
        Certitude : <strong style="color:white;">{conf_eff:.1f} %</strong></div>
      <div class="bar-wrap" style="margin-bottom:16px;">
        <div class="{'bar-fill-red' if fractured_eff else 'bar-fill-green'}"
             style="width:{conf_eff}%;"></div>
      </div>

      <div style="font-size:11px;color:#94a3b8;text-transform:uppercase;
                  margin-bottom:8px;">Zones identifiées</div>
      {det_html}

      <a href="/" class="btn-primary">🔄 Nouvelle analyse</a>
    </div>
  </div>
</div>
</body></html>"""


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  🦴  AI Medical Imaging — EfficientNet B0 + YOLOv8")
    print("="*60)
    print("\n📍 Ouvrez : http://localhost:8000\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
