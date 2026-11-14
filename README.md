# 🦴 Détection de Fractures - AI System

Application web de détection de fractures osseuses utilisant le Deep Learning.

## 📊 Modèle

- **Architecture:** EfficientNet B0
- **Précision:** 83.05%
- **Dataset:** 4,085 radiographies (FracAtlas)

## 🚀 Installation

```bash
pip install -r requirements.txt
```

## ▶️ Lancement

```bash
python app.py
```

Ouvrir: http://localhost:8000

## 📁 Structure

```
├── app.py              # Application FastAPI
├── models/
│   └── best_model.pth  # Modèle entraîné
├── temp/               # Images temporaires
└── requirements.txt    # Dépendances
```

## 💡 Utilisation

1. Sélectionner une radiographie
2. Cliquer sur "ANALYSER"
3. Voir le résultat (Fracture OUI/NON)

## 🛠️ Technologies

- PyTorch
- FastAPI
- EfficientNet B0
