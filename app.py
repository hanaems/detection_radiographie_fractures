"""
Application Simple - Détection de Fractures
"""
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import uvicorn

class BoneFractureClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.efficientnet_b0(weights=None)
        self.features = backbone.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

app = FastAPI()

print("📂 Chargement du modèle...")
model = BoneFractureClassifier()
checkpoint = torch.load('models/best_model.pth', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print("✅ Modèle chargé!")

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Medical Imaging - Détection de Fractures</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: #0a0e27;
            color: #e4e8f0;
            min-height: 100vh;
        }
        
        .header {
            background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
            padding: 20px 40px;
            box-shadow: 0 4px 20px rgba(59, 130, 246, 0.3);
        }
        
        .header-content {
            max-width: 1400px;
            margin: 0 auto;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .logo {
            display: flex;
            align-items: center;
            gap: 12px;
        }
        
        .logo-icon {
            width: 40px;
            height: 40px;
            background: white;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 24px;
        }
        
        .logo-text h1 {
            font-size: 20px;
            font-weight: 600;
            color: white;
        }
        
        .logo-text p {
            font-size: 12px;
            color: rgba(255,255,255,0.8);
            font-weight: 400;
        }
        
        .header-info {
            display: flex;
            gap: 30px;
            font-size: 14px;
        }
        
        .header-info div {
            display: flex;
            flex-direction: column;
            align-items: flex-end;
        }
        
        .header-info strong {
            color: white;
            font-weight: 600;
        }
        
        .header-info span {
            color: rgba(255,255,255,0.7);
            font-size: 12px;
        }
        
        .main-container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 30px 40px;
        }
        
        .patient-card {
            background: #151b3d;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 24px;
            border: 1px solid rgba(59, 130, 246, 0.2);
        }
        
        .patient-info {
            display: flex;
            gap: 40px;
            font-size: 14px;
        }
        
        .patient-info div {
            display: flex;
            flex-direction: column;
            gap: 4px;
        }
        
        .patient-info strong {
            color: #94a3b8;
            font-weight: 500;
            font-size: 12px;
            text-transform: uppercase;
        }
        
        .patient-info span {
            color: #e4e8f0;
            font-weight: 600;
        }
        
        .analysis-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 24px;
        }
        
        .card {
            background: #151b3d;
            border-radius: 12px;
            padding: 24px;
            border: 1px solid rgba(59, 130, 246, 0.2);
        }
        
        .card-header {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 20px;
            padding-bottom: 16px;
            border-bottom: 1px solid rgba(59, 130, 246, 0.2);
        }
        
        .card-icon {
            width: 36px;
            height: 36px;
            background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 20px;
        }
        
        .card-title {
            font-size: 16px;
            font-weight: 600;
            color: white;
        }
        
        .upload-area {
            border: 2px dashed rgba(59, 130, 246, 0.3);
            border-radius: 12px;
            padding: 60px 20px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s;
            background: rgba(59, 130, 246, 0.05);
        }
        
        .upload-area:hover {
            border-color: #3b82f6;
            background: rgba(59, 130, 246, 0.1);
        }
        
        .upload-icon {
            font-size: 48px;
            margin-bottom: 16px;
        }
        
        .upload-text {
            font-size: 16px;
            color: #94a3b8;
            margin-bottom: 8px;
        }
        
        .upload-text strong {
            color: #3b82f6;
        }
        
        .upload-hint {
            font-size: 13px;
            color: #64748b;
        }
        
        input[type="file"] {
            display: none;
        }
        
        #preview {
            text-align: center;
            margin-top: 20px;
        }
        
        #preview img {
            max-width: 100%;
            border-radius: 12px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.4);
        }
        
        .analyze-btn {
            background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
            color: white;
            border: none;
            padding: 16px 32px;
            border-radius: 10px;
            cursor: pointer;
            font-size: 16px;
            font-weight: 600;
            width: 100%;
            margin-top: 24px;
            transition: all 0.3s;
            box-shadow: 0 4px 16px rgba(59, 130, 246, 0.3);
        }
        
        .analyze-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 24px rgba(59, 130, 246, 0.4);
        }
        
        .info-panel {
            background: rgba(59, 130, 246, 0.1);
            border: 1px solid rgba(59, 130, 246, 0.3);
            border-radius: 12px;
            padding: 20px;
        }
        
        .info-item {
            display: flex;
            justify-content: space-between;
            padding: 12px 0;
            border-bottom: 1px solid rgba(59, 130, 246, 0.2);
        }
        
        .info-item:last-child {
            border-bottom: none;
        }
        
        .info-label {
            color: #94a3b8;
            font-size: 14px;
        }
        
        .info-value {
            color: white;
            font-weight: 600;
            font-size: 14px;
        }
        
        .model-badge {
            display: inline-block;
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            color: white;
            padding: 6px 12px;
            border-radius: 6px;
            font-size: 12px;
            font-weight: 600;
        }
        
        @media (max-width: 1024px) {
            .analysis-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="header">
        <div class="header-content">
            <div class="logo">
                <div class="logo-icon">🦴</div>
                <div class="logo-text">
                    <h1>AI Medical Imaging</h1>
                    <p>Système de Détection Automatique de Fractures</p>
                </div>
            </div>
            <div class="header-info">
                <div>
                    <strong>EfficientNet B0</strong>
                    <span>Deep Learning Model</span>
                </div>
                <div>
                    <strong>83.05%</strong>
                    <span>Précision Validation</span>
                </div>
            </div>
        </div>
    </div>
    
    <div class="main-container">
        <div class="patient-card">
            <div class="patient-info">
                <div>
                    <strong>Examen</strong>
                    <span>Radiographie osseuse</span>
                </div>
                <div>
                    <strong>Type d'analyse</strong>
                    <span>Détection de fracture (IA)</span>
                </div>
                <div>
                    <strong>Modèle</strong>
                    <span class="model-badge">EfficientNet B0</span>
                </div>
            </div>
        </div>
        
        <form method="post" action="/predict" enctype="multipart/form-data">
            <div class="analysis-grid">
                <div class="card">
                    <div class="card-header">
                        <div class="card-icon">📸</div>
                        <div class="card-title">Image radiographique</div>
                    </div>
                    
                    <div class="upload-area" onclick="document.getElementById('fileInput').click()">
                        <div class="upload-icon">⬆️</div>
                        <div class="upload-text">
                            <strong>Cliquez pour sélectionner</strong> ou glissez une image
                        </div>
                        <div class="upload-hint">Formats acceptés: JPG, PNG, JPEG</div>
                    </div>
                    
                    <input type="file" name="file" id="fileInput" accept="image/*" required onchange="previewImage()">
                    
                    <div id="preview"></div>
                    
                    <button type="submit" class="analyze-btn">
                        🔍 LANCER L'ANALYSE IA
                    </button>
                </div>
                
                <div class="card">
                    <div class="card-header">
                        <div class="card-icon">ℹ️</div>
                        <div class="card-title">Informations du modèle</div>
                    </div>
                    
                    <div class="info-panel">
                        <div class="info-item">
                            <span class="info-label">Architecture</span>
                            <span class="info-value">EfficientNet B0</span>
                        </div>
                        <div class="info-item">
                            <span class="info-label">Dataset d'entraînement</span>
                            <span class="info-value">FracAtlas (4,085 images)</span>
                        </div>
                        <div class="info-item">
                            <span class="info-label">Précision validation</span>
                            <span class="info-value">83.05%</span>
                        </div>
                        <div class="info-item">
                            <span class="info-label">Classes détectées</span>
                            <span class="info-value">2 (Fracture / Normal)</span>
                        </div>
                        <div class="info-item">
                            <span class="info-label">Temps d'analyse</span>
                            <span class="info-value">< 1 seconde</span>
                        </div>
                        <div class="info-item">
                            <span class="info-label">Framework</span>
                            <span class="info-value">PyTorch 2.0</span>
                        </div>
                    </div>
                    
                    <div style="margin-top: 24px; padding: 16px; background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 8px;">
                        <div style="font-size: 13px; color: #10b981; font-weight: 500;">
                            ✓ Modèle validé et prêt pour l'analyse
                        </div>
                    </div>
                </div>
            </div>
        </form>
    </div>

    <script>
        function previewImage() {
            const file = document.getElementById('fileInput').files[0];
            const preview = document.getElementById('preview');
            
            if (file) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    preview.innerHTML = '<img src="' + e.target.result + '" alt="Radiographie">';
                }
                reader.readAsDataURL(file);
            }
        }
    </script>
</body>
</html>
    """

import base64

@app.post("/predict", response_class=HTMLResponse)
async def predict(file: UploadFile = File(...)):
    # Lire l'image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')
    
    # Convertir image en base64 pour l'afficher
    img_base64 = base64.b64encode(contents).decode('utf-8')
    
    # Transformer et prédire
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.nn.functional.softmax(output, dim=1).squeeze()
    
    pred_idx = int(torch.argmax(probs))
    classes = ['Fractured', 'Not_Fractured']
    prediction = classes[pred_idx]
    confidence = float(probs[pred_idx]) * 100
    prob_fractured = float(probs[0]) * 100
    prob_not_fractured = float(probs[1]) * 100
    
    # Résultat HTML
    result_class = "fractured" if prediction == "Fractured" else "not-fractured"
    result_text = "🦴 FRACTURE DÉTECTÉE" if prediction == "Fractured" else "✅ PAS DE FRACTURE"
    
    return f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Résultat de l'analyse - AI Medical Imaging</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        
        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: #0a0e27;
            color: #e4e8f0;
            min-height: 100vh;
        }}
        
        .header {{
            background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
            padding: 20px 40px;
            box-shadow: 0 4px 20px rgba(59, 130, 246, 0.3);
        }}
        
        .header-content {{
            max-width: 1400px;
            margin: 0 auto;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        
        .logo {{
            display: flex;
            align-items: center;
            gap: 12px;
        }}
        
        .logo-icon {{
            width: 40px;
            height: 40px;
            background: white;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 24px;
        }}
        
        .logo-text h1 {{
            font-size: 20px;
            font-weight: 600;
            color: white;
        }}
        
        .logo-text p {{
            font-size: 12px;
            color: rgba(255,255,255,0.8);
        }}
        
        .main-container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 30px 40px;
        }}
        
        .result-header {{
            background: {'linear-gradient(135deg, #dc2626 0%, #991b1b 100%)' if prediction == 'Fractured' else 'linear-gradient(135deg, #10b981 0%, #059669 100%)'};
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 24px;
            display: flex;
            align-items: center;
            gap: 20px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        }}
        
        .result-icon {{
            font-size: 48px;
        }}
        
        .result-text h2 {{
            font-size: 28px;
            font-weight: 700;
            color: white;
            margin-bottom: 8px;
        }}
        
        .result-text p {{
            font-size: 16px;
            color: rgba(255,255,255,0.9);
        }}
        
        .analysis-grid {{
            display: grid;
            grid-template-columns: 1.2fr 1fr;
            gap: 24px;
            margin-bottom: 24px;
        }}
        
        .card {{
            background: #151b3d;
            border-radius: 12px;
            padding: 24px;
            border: 1px solid rgba(59, 130, 246, 0.2);
        }}
        
        .card-header {{
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 20px;
            padding-bottom: 16px;
            border-bottom: 1px solid rgba(59, 130, 246, 0.2);
        }}
        
        .card-icon {{
            width: 36px;
            height: 36px;
            background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 20px;
        }}
        
        .card-title {{
            font-size: 16px;
            font-weight: 600;
            color: white;
        }}
        
        .image-container {{
            text-align: center;
            background: #0a0e27;
            border-radius: 12px;
            padding: 20px;
        }}
        
        .image-container img {{
            max-width: 100%;
            border-radius: 12px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.4);
        }}
        
        .metrics-grid {{
            display: grid;
            gap: 16px;
        }}
        
        .metric-card {{
            background: rgba(59, 130, 246, 0.1);
            border: 1px solid rgba(59, 130, 246, 0.3);
            border-radius: 10px;
            padding: 20px;
        }}
        
        .metric-label {{
            font-size: 13px;
            color: #94a3b8;
            margin-bottom: 8px;
            text-transform: uppercase;
            font-weight: 500;
        }}
        
        .metric-value {{
            font-size: 32px;
            font-weight: 700;
            color: white;
        }}
        
        .metric-unit {{
            font-size: 18px;
            color: #94a3b8;
        }}
        
        .probability-bars {{
            margin-top: 20px;
        }}
        
        .prob-item {{
            margin-bottom: 16px;
        }}
        
        .prob-header {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 8px;
            font-size: 14px;
        }}
        
        .prob-label {{
            color: #94a3b8;
            font-weight: 500;
        }}
        
        .prob-value {{
            color: white;
            font-weight: 700;
        }}
        
        .prob-bar {{
            height: 8px;
            background: rgba(59, 130, 246, 0.2);
            border-radius: 4px;
            overflow: hidden;
        }}
        
        .prob-fill {{
            height: 100%;
            background: linear-gradient(90deg, #3b82f6 0%, #2563eb 100%);
            border-radius: 4px;
            transition: width 1s ease;
        }}
        
        .action-btn {{
            background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
            color: white;
            border: none;
            padding: 16px 32px;
            border-radius: 10px;
            cursor: pointer;
            font-size: 16px;
            font-weight: 600;
            width: 100%;
            transition: all 0.3s;
            box-shadow: 0 4px 16px rgba(59, 130, 246, 0.3);
        }}
        
        .action-btn:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 24px rgba(59, 130, 246, 0.4);
        }}
        
        .confidence-indicator {{
            display: inline-flex;
            align-items: center;
            gap: 8px;
            background: rgba(59, 130, 246, 0.2);
            padding: 8px 16px;
            border-radius: 8px;
            font-size: 14px;
            font-weight: 600;
        }}
        
        @media (max-width: 1024px) {{
            .analysis-grid {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <div class="header-content">
            <div class="logo">
                <div class="logo-icon">🦴</div>
                <div class="logo-text">
                    <h1>AI Medical Imaging</h1>
                    <p>Résultats de l'analyse IA</p>
                </div>
            </div>
            <div class="confidence-indicator">
                <span>Confiance: {confidence:.1f}%</span>
            </div>
        </div>
    </div>
    
    <div class="main-container">
        <div class="result-header">
            <div class="result-icon">{'🚨' if prediction == 'Fractured' else '✅'}</div>
            <div class="result-text">
                <h2>{'FRACTURE DÉTECTÉE' if prediction == 'Fractured' else 'AUCUNE FRACTURE DÉTECTÉE'}</h2>
                <p>Analyse complétée avec succès par le modèle EfficientNet B0</p>
            </div>
        </div>
        
        <div class="analysis-grid">
            <div class="card">
                <div class="card-header">
                    <div class="card-icon">📸</div>
                    <div class="card-title">Radiographie analysée</div>
                </div>
                <div class="image-container">
                    <img src="data:image/jpeg;base64,{img_base64}" alt="Radiographie">
                </div>
            </div>
            
            <div class="card">
                <div class="card-header">
                    <div class="card-icon">📊</div>
                    <div class="card-title">Métriques de l'analyse</div>
                </div>
                
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-label">Prédiction</div>
                        <div class="metric-value">
                            {'Fracture' if prediction == 'Fractured' else 'Normal'}
                        </div>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-label">Niveau de confiance</div>
                        <div class="metric-value">
                            {confidence:.1f}<span class="metric-unit">%</span>
                        </div>
                    </div>
                </div>
                
                <div class="probability-bars">
                    <div class="prob-item">
                        <div class="prob-header">
                            <span class="prob-label">🦴 Probabilité Fracture</span>
                            <span class="prob-value">{prob_fractured:.1f}%</span>
                        </div>
                        <div class="prob-bar">
                            <div class="prob-fill" style="width: {prob_fractured}%"></div>
                        </div>
                    </div>
                    
                    <div class="prob-item">
                        <div class="prob-header">
                            <span class="prob-label">✅ Probabilité Normal</span>
                            <span class="prob-value">{prob_not_fractured:.1f}%</span>
                        </div>
                        <div class="prob-bar">
                            <div class="prob-fill" style="width: {prob_not_fractured}%"></div>
                        </div>
                    </div>
                </div>
                
                <button class="action-btn" onclick="window.location.href='/'">
                    🔄 NOUVELLE ANALYSE
                </button>
            </div>
        </div>
    </div>
</body>
</html>
    """

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  🦴 DÉTECTION DE FRACTURES - VERSION SIMPLE")
    print("="*60)
    print("\n📍 Ouvrez: http://localhost:8000")
    print("\n" + "="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
