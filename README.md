# 🎯 Prédiction de Churn Client - Télécoms

> Identifier les clients à risque 3 mois avant leur départ pour lancer des actions de rétention.

## 📊 Problématique Business

**Contexte** : L'entreprise de télécommunications perd **25% de ses clients chaque année**.

**Coût** : Le coût d'acquisition d'un nouveau client (CAC) est **5x plus élevé** que la rétention.

**Objectif** : Identifier les clients à risque **3 mois avant** leur départ pour lancer des campagnes de rétention ciblées.

**Impact attendu** :
- Réduction du churn de 25% → 18% (-28%)
- Économie annuelle : 2M€ (10,000 clients × 200€/client)
- ROI campagne rétention : 5:1

---

## 🗂️ Dataset

**Source** : [Telco Customer Churn - IBM](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

**Volume** : 7,043 clients avec 21 features

**Features clés** :
- Démographie : `SeniorCitizen`, `Partner`, `Dependents`
- Services : `PhoneService`, `InternetService`, `StreamingTV`
- Contrat : `Contract`, `MonthlyCharges`, `TotalCharges`, `tenure`
- Support : `TechSupport`, `OnlineBackup`
- Target : `Churn` (Yes/No)

**Téléchargement** :
```bash
# Option 1 : Kaggle CLI (nécessite API key)
kaggle datasets download -d blastchar/telco-customer-churn

# Option 2 : Téléchargement manuel
# https://www.kaggle.com/datasets/blastchar/telco-customer-churn
# Placer le CSV dans data/raw/
```

---

## 🏗️ Architecture Technique

```
┌──────────────┐    ┌─────────────────┐    ┌────────────┐    ┌─────────┐    ┌───────────┐
│ Data Source  │───▶│ Feature Eng.    │───▶│ ML Pipeline│───▶│   API   │───▶│ Dashboard │
│  (Kaggle)    │    │   (pandas)      │    │ (sklearn)  │    │(FastAPI)│    │(Streamlit)│
└──────────────┘    └─────────────────┘    └────────────┘    └─────────┘    └───────────┘
    CSV                 numpy                 XGBoost          Docker         Plotly
```

---

## 📦 Structure du Projet

```
01-churn-prediction-telecom/
├── README.md                     # Ce fichier
├── requirements.txt              # Dépendances Python
├── .gitignore                    # Fichiers à ignorer
├── Dockerfile                    # Container API
├── docker-compose.yml            # Orchestration
│
├── data/
│   ├── raw/                      # Données brutes (non versionnées)
│   ├── processed/                # Données nettoyées
│   └── README.md                 # Instructions téléchargement
│
├── notebooks/
│   ├── 01_eda.ipynb              # Analyse exploratoire
│   ├── 02_feature_engineering.ipynb
│   └── 03_modeling.ipynb         # Entraînement modèle
│
├── src/
│   ├── __init__.py
│   ├── config.py                 # Configuration
│   ├── data_loader.py            # Chargement données
│   ├── feature_engineering.py    # Features custom
│   ├── model_trainer.py          # Entraînement
│   └── predictor.py              # Inférence
│
├── api/
│   ├── main.py                   # FastAPI app
│   ├── schemas.py                # Pydantic models
│   └── routers/
│       ├── predict.py            # Endpoint /predict
│       └── health.py             # Endpoint /health
│
├── dashboards/
│   └── streamlit_app.py          # Dashboard interactif
│
├── models/
│   └── .gitkeep                  # Modèles sauvegardés (MLflow)
│
└── tests/
    ├── test_feature_engineering.py
    └── test_api.py
```

---

## 🚀 Quick Start

### 1. Installation

```bash
# Cloner le template
git clone <repo-url>
cd 01-churn-prediction-telecom

# Créer environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Installer dépendances
pip install -r requirements.txt
```

### 2. Télécharger les données

```bash
# Suivre les instructions dans data/README.md
# Placer telco_churn.csv dans data/raw/
```

### 3. Analyse exploratoire

```bash
jupyter notebook notebooks/01_eda.ipynb
```

### 4. Entraîner le modèle

```bash
python src/model_trainer.py
# Le modèle sera sauvegardé dans models/
```

### 5. Lancer l'API

```bash
# Option 1 : Local
uvicorn api.main:app --reload --port 8000

# Option 2 : Docker
docker-compose up --build
```

### 6. Tester l'API

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "tenure": 12,
    "MonthlyCharges": 70.5,
    "TotalCharges": 846.0,
    "Contract": "Month-to-month",
    "InternetService": "Fiber optic"
  }'
```

### 7. Lancer le dashboard

```bash
streamlit run dashboards/streamlit_app.py
```

---

## 🎯 Critères de Réussite

| Métrique | Objectif | Justification |
|----------|----------|---------------|
| **Recall Churn** | > 80% | Minimiser clients perdus non détectés (faux négatifs) |
| **Precision** | > 70% | Limiter fausses alertes (coût campagnes inutiles) |
| **API Latence** | < 100ms | Temps réel pour intégration CRM |
| **F1-Score** | > 0.75 | Équilibre recall/precision |

---

## 🛠️ Stack Technique

**Machine Learning** :
- Python 3.10+
- pandas, numpy
- scikit-learn 1.3+
- XGBoost 2.0+
- SHAP (explainability)

**MLOps** :
- MLflow (tracking, registry)
- Docker (containerisation)

**API** :
- FastAPI 0.100+
- uvicorn (server ASGI)
- pydantic (validation)

**Visualisation** :
- Streamlit 1.28+
- plotly
- seaborn, matplotlib

**Dev Tools** :
- Jupyter Notebook
- pytest (tests)
- black (formatting)
- Git + GitHub

---

## 💡 Features Engineering Clés

**Ratios** :
```python
# Ratio prix/ancienneté (indicateur augmentation)
df['price_increase_ratio'] = df['MonthlyCharges'] / (df['tenure'] + 1)

# Engagement services (nombre de services souscrits)
service_cols = ['PhoneService', 'InternetService', 'OnlineBackup', 'StreamingTV']
df['service_engagement'] = df[service_cols].apply(lambda x: x.str.contains('Yes').sum(), axis=1)
```

**Délais** :
```python
# Délai depuis dernier contact support (simulation)
df['days_since_support'] = np.random.randint(0, 365, len(df))
```

**Segmentation** :
```python
# Segmentation ancienneté
df['tenure_segment'] = pd.cut(df['tenure'], bins=[0, 12, 24, 60, 100],
                               labels=['<1yr', '1-2yr', '2-5yr', '>5yr'])
```

---

## 🔧 Gestion du Déséquilibre

Le dataset a un déséquilibre (26.5% churn vs 73.5% non-churn).

**Techniques** :
```python
# Option 1 : Class weight
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(class_weight='balanced')

# Option 2 : SMOTE (oversampling)
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Option 3 : Focal Loss (pour deep learning)
# Penalise davantage les erreurs sur classe minoritaire
```

---

## 📊 Métriques Business

**Calcul ROI campagne rétention** :
```python
# Hypothèses
n_clients_at_risk = 1000
true_churn_rate = 0.80  # Parmi ceux prédits
retention_campaign_success = 0.30  # 30% retenus
customer_ltv = 200  # Lifetime value moyenne
campaign_cost_per_client = 20

# Calcul
clients_saved = n_clients_at_risk * true_churn_rate * retention_campaign_success
revenue_saved = clients_saved * customer_ltv
campaign_cost = n_clients_at_risk * campaign_cost_per_client
roi = (revenue_saved - campaign_cost) / campaign_cost

print(f"Clients sauvés : {clients_saved}")
print(f"Revenu sauvé : {revenue_saved}€")
print(f"ROI : {roi:.1%}")  # 5:1 attendu
```

---

## 🚢 Déploiement

### Docker

```bash
# Build image
docker build -t churn-api .

# Run container
docker run -p 8000:8000 churn-api
```

### Streamlit Cloud

```bash
# 1. Push sur GitHub
git add .
git commit -m "feat: churn prediction app"
git push origin main

# 2. Déployer sur https://streamlit.io/cloud
# Connecter repo GitHub
# Sélectionner dashboards/streamlit_app.py
```

---

## 📚 Ressources

**Datasets** :
- [Telco Customer Churn - Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

**Articles** :
- [Survival Analysis for Churn Prediction](https://towardsdatascience.com/survival-analysis-churn-prediction)
- [SHAP for Model Explainability](https://shap.readthedocs.io/)

**Benchmarks** :
- State-of-the-art : F1 = 0.82 (XGBoost + feature engineering)
- Baseline : F1 = 0.65 (Logistic Regression)

---

## 📝 TODO

- [ ] Télécharger dataset Kaggle
- [ ] Compléter notebooks EDA
- [ ] Implémenter feature engineering
- [ ] Entraîner modèle XGBoost
- [ ] Atteindre Recall > 80%
- [ ] Créer API FastAPI
- [ ] Tester latence < 100ms
- [ ] Dashboard Streamlit
- [ ] SHAP values pour explainability
- [ ] Dockeriser l'application
- [ ] Déployer sur Streamlit Cloud
- [ ] Écrire README final avec résultats

---

## 👤 Auteur

**Votre Nom**
- GitHub : [@votre-username](https://github.com/votre-username)
- LinkedIn : [Votre Profil](https://linkedin.com/in/votre-profil)
- Portfolio : [votre-site.com](https://votre-site.com)

---

## 📄 Licence

MIT License - Libre d'utilisation pour votre portfolio.
