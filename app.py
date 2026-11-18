"""
FractureAI - API Backend Complète
Auth + Analyse IA (EfficientNet B0 + YOLOv8)
"""
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session
from datetime import datetime
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageDraw
from ultralytics import YOLO
import io, base64, uvicorn

# Import auth & database
from database import get_db, User
from auth import hash_password, verify_password, create_access_token

# ══════════════════════════════════════════════════════════════════════════════
# MODÈLES IA
# ══════════════════════════════════════════════════════════════════════════════
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
        x = self.features(x)
        x = self.avgpool(x)
        return self.classifier(torch.flatten(x, 1))

# ══════════════════════════════════════════════════════════════════════════════
# FASTAPI APP
# ══════════════════════════════════════════════════════════════════════════════
app = FastAPI(title="FractureAI - Full API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ══════════════════════════════════════════════════════════════════════════════
# MODÈLES PYDANTIC POUR AUTH
# ══════════════════════════════════════════════════════════════════════════════
class RegisterRequest(BaseModel):
    full_name: str
    email: EmailStr
    password: str
    phone: str = None
    speciality: str = None
    address: str = None
    rpps: str = None

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class ProfileUpdate(BaseModel):
    full_name: str = None
    phone: str = None
    speciality: str = None
    address: str = None
    rpps: str = None
    photo: str = None

class PasswordChange(BaseModel):
    current_password: str
    new_password: str

class AuthResponse(BaseModel):
    token: str
    user: dict

# ══════════════════════════════════════════════════════════════════════════════
# ROUTES AUTH
# ══════════════════════════════════════════════════════════════════════════════
@app.post("/api/auth/register", response_model=AuthResponse)
def register(data: RegisterRequest, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.email == data.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email déjà utilisé")
    
    names = data.full_name.split(' ', 1)
    first_name = names[0]
    last_name = names[1] if len(names) > 1 else ""
    
    user = User(
        username=data.email.split('@')[0],
        email=data.email,
        password_hash=hash_password(data.password),
        first_name=first_name,
        last_name=last_name,
        phone=data.phone,
        speciality=data.speciality,
        address=data.address,
        rpps=data.rpps
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    
    token = create_access_token({"sub": str(user.id), "email": user.email})
    
    return {
        "token": token,
        "user": {
            "id": user.id,
            "email": user.email,
            "first_name": user.first_name,
            "last_name": user.last_name,
            "phone": user.phone,
            "speciality": user.speciality,
            "address": user.address,
            "rpps": user.rpps
        }
    }

@app.post("/api/auth/login", response_model=AuthResponse)
def login(data: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == data.email).first()
    if not user:
        raise HTTPException(status_code=401, detail="Email ou mot de passe incorrect")
    
    if not verify_password(data.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Email ou mot de passe incorrect")
    
    user.last_login = datetime.utcnow()
    db.commit()
    
    token = create_access_token({"sub": str(user.id), "email": user.email})
    
    return {
        "token": token,
        "user": {
            "id": user.id,
            "email": user.email,
            "first_name": user.first_name,
            "last_name": user.last_name
        }
    }

@app.get("/api/profile/{user_id}")
def get_profile(user_id: int, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="Utilisateur non trouvé")
    
    return {
        "id": user.id,
        "full_name": f"{user.first_name} {user.last_name}".strip(),
        "email": user.email,
        "phone": user.phone,
        "speciality": user.speciality,
        "address": user.address,
        "rpps": user.rpps,
        "photo": user.photo
    }

@app.put("/api/profile/{user_id}")
def update_profile(user_id: int, data: ProfileUpdate, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="Utilisateur non trouvé")
    
    if data.full_name:
        names = data.full_name.strip().split(' ', 1)
        user.first_name = names[0]
        user.last_name = names[1] if len(names) > 1 else ""
    if data.phone:
        user.phone = data.phone.strip()
    if data.speciality:
        user.speciality = data.speciality.strip()
    if data.address:
        user.address = data.address.strip()
    if data.rpps:
        user.rpps = data.rpps.strip()
    if data.photo:
        user.photo = data.photo
    
    db.commit()
    db.refresh(user)
    
    return {
        "message": "Profil mis à jour",
        "user": {
            "id": user.id,
            "full_name": f"{user.first_name} {user.last_name}".strip(),
            "email": user.email,
            "phone": user.phone,
            "speciality": user.speciality,
            "address": user.address,
            "rpps": user.rpps,
            "photo": user.photo
        }
    }

@app.put("/api/profile/{user_id}/password")
def change_password(user_id: int, data: PasswordChange, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="Utilisateur non trouvé")
    
    if not verify_password(data.current_password, user.password_hash):
        raise HTTPException(status_code=401, detail="Mot de passe actuel incorrect")
    
    user.password_hash = hash_password(data.new_password)
    db.commit()
    
    return {"message": "Mot de passe changé avec succès"}

# ══════════════════════════════════════════════════════════════════════════════
# CHARGEMENT DES MODÈLES IA
# ══════════════════════════════════════════════════════════════════════════════
print("📂 Chargement des modèles IA...")
eff_model = BoneFractureClassifier()
ck = torch.load("models/best_model.pth", map_location="cpu")
eff_model.load_state_dict(ck["model_state_dict"])
eff_model.eval()
print("✅ EfficientNet chargé!")

yolo_model = YOLO("models/best.pt")
print("✅ YOLOv8 chargé!")

tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

YOLO_COLORS = [
    (255,80,80),(80,200,120),(80,150,255),
    (255,180,50),(200,80,255),(50,220,220),(255,100,180),
]

# ══════════════════════════════════════════════════════════════════════════════
# FONCTIONS D'ANALYSE
# ══════════════════════════════════════════════════════════════════════════════
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

# ══════════════════════════════════════════════════════════════════════════════
# API ANALYSE POUR ANGULAR
# ══════════════════════════════════════════════════════════════════════════════
@app.post("/api/analyze")
async def api_analyze(file: UploadFile = File(...)):
    """Endpoint API pour l'analyse depuis le dashboard Angular."""
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    
    # Lancer les deux analyses
    fractured_eff, conf_eff, p_frac, p_norm = run_efficientnet(image)
    detections, has_fracture_yolo, img_yolo_b64 = run_yolo(image)
    
    # Encoder l'image originale
    img_orig_b64 = base64.b64encode(contents).decode()
    
    return {
        "diagnosis": "Fracture détectée" if fractured_eff else "Aucune fracture",
        "confidence": round(conf_eff, 1),
        "fractured": fractured_eff,
        "efficientnet": {
            "prediction": "Fracture" if fractured_eff else "Normal",
            "confidence": round(conf_eff, 1),
            "prob_fracture": round(p_frac, 1),
            "prob_normal": round(p_norm, 1)
        },
        "yolo": {
            "detections": len(detections),
            "has_fracture": has_fracture_yolo,
            "zones": [
                {
                    "label": d["label"],
                    "confidence": round(d["conf"], 1),
                    "color": f"rgb({d['color'][0]},{d['color'][1]},{d['color'][2]})"
                }
                for d in detections
            ]
        },
        "images": {
            "original": f"data:image/jpeg;base64,{img_orig_b64}",
            "annotated": f"data:image/jpeg;base64,{img_yolo_b64}"
        },
        "agreement": fractured_eff == has_fracture_yolo
    }

# ══════════════════════════════════════════════════════════════════════════════
# LANCEMENT
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  🦴  FractureAI - API Complète")
    print("="*60)
    print("\n📍 API disponible sur: http://localhost:8000")
    print("   - Auth: /api/auth/login, /api/auth/register")
    print("   - Profile: /api/profile/{user_id}")
    print("   - Analyse IA: /api/analyze\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
