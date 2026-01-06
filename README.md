# 📊 Prédiction de la Value-at-Risk : Apprentissage Profond vs Modèles Statistiques

> Comparaison des modèles d'apprentissage profond (ANN, LSTM) et statistiques (ARIMA, SARIMA) pour la prédiction de la Value-at-Risk sur les indices boursiers MENA.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📑 Table des Matières

- [À Propos](#à-propos)
- [Fonctionnalités](#fonctionnalités)
- [Architecture du Projet](#architecture-du-projet)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Données](#données)
- [Modèles Implémentés](#modèles-implémentés)
- [Résultats](#résultats)
- [Équipe](#équipe)
- [Références](#références)
- [License](#license)

## 🎯 À Propos

Ce projet académique compare l'efficacité des **modèles d'apprentissage profond** (Deep Learning) et des **modèles statistiques classiques** pour la prédiction de la **Value-at-Risk (VaR)** sur les marchés boursiers MENA (Middle East and North Africa).

### Contexte

La Value-at-Risk est une mesure de risque financier largement utilisée qui quantifie la perte maximale potentielle sur un horizon temporel donné à un niveau de confiance spécifique. Ce projet explore si les techniques modernes d'apprentissage profond peuvent améliorer les prédictions de VaR par rapport aux méthodes statistiques traditionnelles.

### Objectifs

- ✅ Implémenter et comparer 4 modèles de prédiction
- ✅ Évaluer la VaR à l'aide de la Simulation Historique Bootstrap
- ✅ Valider les résultats par backtesting rigoureux
- ✅ Analyser les performances sur les marchés MENA

## ⚡ Fonctionnalités

- **4 Modèles de Prédiction**
  - 🧠 ANN (Artificial Neural Network)
  - 🔄 LSTM (Long Short-Term Memory)
  - 📈 ARIMA (AutoRegressive Integrated Moving Average)
  - 📊 SARIMA (Seasonal ARIMA)

- **Calcul Avancé de VaR**
  - Simulation Historique Bootstrap (10,000 échantillons)
  - Niveaux de confiance multiples (95%, 99%)
  - Validation par backtesting

- **Analyse Complète**
  - Métriques de performance (MAE, RMSE)
  - Taux de violations VaR
  - Visualisations détaillées
  - Tableau de bord de validation automatique

## 📁 Architecture du Projet

```
var-prediction-project/
│
├── 📓 Projet_VaR_Presentation.ipynb    # Notebook principal (Français)
├── 📄 README.md                         # Ce fichier
├── 📋 requirements.txt                  # Dépendances Python
│
├── 📂 data/
│   ├── ADI.csv                          # Abu Dhabi Securities Market
│   ├── MASI.csv                         # Moroccan All Shares Index
│   ├── TASI.csv                         # Tadawul All Share Index
│   ├── Tunindex.csv                     # Tunis Stock Exchange
│   ├── CAC40.csv                        # CAC 40 (Benchmark)
│   ├── S&P500.csv                       # S&P 500 (Benchmark)
│   ├── ADITest.csv                      # Données de test
│   ├── MASITest.csv
│   ├── TASITest.csv
│   └── TunindexTest.csv
│
├── 📂 docs/
│   ├── methodology.md                   # Méthodologie détaillée
│   ├── results_analysis.md              # Analyse des résultats
│   └── references.pdf                   # Références académiques
│
└── 📂 outputs/
    ├── figures/                         # Graphiques générés
    ├── models/                          # Modèles sauvegardés
    └── results/                         # Résultats exportés
```

## 🔧 Installation

### Prérequis

- Python 3.8 ou supérieur
- pip (gestionnaire de paquets Python)
- Compte Google Drive (pour Google Colab)

### Option 1 : Google Colab (Recommandé)

1. **Ouvrir le notebook dans Colab**
   ```
   https://colab.research.google.com/
   ```

2. **Télécharger le notebook**
   - Uploadez `Projet_VaR_Presentation.ipynb`

3. **Monter Google Drive**
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

4. **Placer les données**
   - Créer le dossier `/content/drive/MyDrive/data/`
   - Uploader tous les fichiers CSV

### Option 2 : Installation Locale

1. **Cloner le repository**
   ```bash
   git clone https://github.com/votre-username/var-prediction-project.git
   cd var-prediction-project
   ```

2. **Créer un environnement virtuel**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Sur Windows: venv\Scripts\activate
   ```

3. **Installer les dépendances**
   ```bash
   pip install -r requirements.txt
   ```

4. **Lancer Jupyter**
   ```bash
   jupyter notebook Projet_VaR_Presentation.ipynb
   ```

### Dépendances Principales

```txt
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
tensorflow>=2.8.0
scikit-learn>=0.24.0
statsmodels>=0.13.0
scipy>=1.7.0
```

## 🚀 Utilisation

### Exécution Complète

1. **Ouvrir le notebook** `Projet_VaR_Presentation.ipynb`

2. **Configurer les paramètres** (Section 2.0)
   ```python
   # Paramètres du Modèle
   LOOKBACK = 10
   RANDOM_SEED = 42
   
   # Paramètres ANN/LSTM
   ANN_EPOCHS = 200
   LSTM_EPOCHS = 200
   
   # Paramètres VaR
   N_BOOTSTRAP = 10000
   CONFIDENCE_LEVELS = [0.95, 0.99]
   ```

3. **Exécuter toutes les cellules**
   - Dans Jupyter: `Cell → Run All`
   - Dans Colab: `Runtime → Run all`

4. **Visualiser les résultats**
   - Section 6: Résultats et analyses
   - Section 7: Conclusions
   - Section 8: Validation automatique

### Exécution Rapide (Mode Test)

Pour un test rapide avec moins d'époques:

```python
# Configuration rapide
ANN_EPOCHS = 50
LSTM_EPOCHS = 50
N_BOOTSTRAP = 1000
SKIP_SARIMA = True  # SARIMA est le plus lent
```

### Personnalisation

#### Ajouter un Nouvel Indice

1. Placer les fichiers CSV dans `/data/`
2. Modifier la configuration:
   ```python
   data_paths['train']['NOUVEAU_INDICE'] = '/path/to/data.csv'
   data_paths['test']['NOUVEAU_INDICE'] = '/path/to/dataTest.csv'
   mena_indices.append('NOUVEAU_INDICE')
   ```

#### Modifier l'Architecture des Modèles

```python
# Section 2.0 - Configuration
ANN_NEURONS = [128, 64, 32]     # Augmenter la capacité
LSTM_NEURONS = [128, 64]        # Plus de neurones
LOOKBACK = 20                   # Fenêtre plus longue
```

## 📊 Données

### Indices MENA

| Indice | Pays | Période | Échantillons |
|--------|------|---------|--------------|
| **ADI** | Émirats Arabes Unis | 2015-2024 | ~2500 |
| **MASI** | Maroc | 2015-2024 | ~2500 |
| **TASI** | Arabie Saoudite | 2015-2024 | ~2500 |
| **Tunindex** | Tunisie | 2015-2024 | ~2500 |

### Indices de Référence

| Indice | Région | Usage |
|--------|--------|-------|
| **CAC40** | Europe | Benchmark développé |
| **S&P500** | USA | Benchmark mondial |

### Format des Données

```csv
Date,Price,Open,High,Low,Vol.,Change %
Dec 31, 2014,4528.93,4441.67,4554.09,4424.24,213.88M,1.91%
Dec 30, 2014,4444.03,4543.14,4543.14,4416.47,178.16M,-2.18%
...
```

## 🤖 Modèles Implémentés

### 1. ANN (Réseau de Neurones Artificiels)

**Architecture:**
```
Input (10) → Dense(64) → Dropout(0.2) → 
Dense(32) → Dropout(0.2) → Dense(16) → 
Dropout(0.1) → Output(1)
```

**Avantages:**
- Capture les relations non-linéaires
- Entraînement rapide
- Bonne généralisation

### 2. LSTM (Long Short-Term Memory)

**Architecture:**
```
Input (10, 1) → LSTM(64) → Dropout(0.2) → 
LSTM(32) → Dropout(0.2) → Dense(16) → Output(1)
```

**Avantages:**
- Mémoire à long terme
- Capture les dépendances temporelles
- Excellent pour les séries temporelles

### 3. ARIMA

**Configuration:**
- Ordre optimal sélectionné par AIC
- Range: p,d,q ∈ [0,3]
- Prédiction rolling window

**Avantages:**
- Interprétable
- Robuste
- Peu de paramètres

### 4. SARIMA

**Configuration:**
- Période saisonnière: 5 jours (hebdomadaire)
- Ordre saisonnier: P,D,Q ∈ [0,2]
- Selection automatique par AIC

**Avantages:**
- Capture la saisonnalité
- Extension naturelle d'ARIMA
- Bonne pour patterns cycliques

## 📈 Résultats

### Performance Moyenne (MAE)

| Modèle | MAE Moyen | Rang |
|--------|-----------|------|
| **LSTM** | 0.005633 | 🥇 1 |
| **ANN** | 0.005688 | 🥈 2 |
| **SARIMA** | 0.005819 | 🥉 3 |
| **ARIMA** | 0.005831 | 4 |

### Estimations VaR (Moyennes)

| Modèle | VaR 95% | VaR 99% |
|--------|---------|---------|
| **ANN** | 0.0125 | 0.0224 |
| **LSTM** | 0.0126 | 0.0226 |
| **ARIMA** | 0.0126 | 0.0224 |
| **SARIMA** | 0.0123 | 0.0219 |

### Backtesting (Taux de Violations)

| Modèle | Violations 95% | Violations 99% | Statut |
|--------|----------------|----------------|--------|
| **ANN** | 5.32% | 1.10% | ✅ Excellent |
| **LSTM** | 5.32% | 1.10% | ✅ Excellent |
| **ARIMA** | 5.32% | 1.10% | ✅ Excellent |
| **SARIMA** | 5.32% | 1.10% | ✅ Excellent |

> 📌 **Note:** Violations attendues = 5.00% pour VaR 95%, 1.00% pour VaR 99%

### Validation Complète

```
✅ TOUTES LES VÉRIFICATIONS RÉUSSIES!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ VaR dans plage réaliste
✓ VaR 99% > VaR 95%
✓ Toutes valeurs positives
✓ Violations acceptables
✓ Couverture adéquate
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
6/6 vérifications réussies (100%)
```

## 🎓 Équipe

Ce projet a été développé par :

- **Aws Ourari** - [GitHub](https://github.com/aws-ourari) | [LinkedIn](https://linkedin.com/in/aws-ourari)
- **Nairi Najla** - [GitHub](https://github.com/nairi-najla) | [LinkedIn](https://linkedin.com/in/nairi-najla)
- **Ines Jaziri** - [GitHub](https://github.com/ines-jaziri) | [LinkedIn](https://linkedin.com/in/ines-jaziri)

> 📧 Pour toute question : [aws.ourari@example.com](mailto:aws.ourari@example.com)

## 📚 Références

### Articles Scientifiques

1. **Kupiec, P. H.** (1995). *Techniques for verifying the accuracy of risk measurement models.* Journal of Derivatives, 3(2), 73-84.

2. **Christoffersen, P. F.** (1998). *Evaluating interval forecasts.* International Economic Review, 39(4), 841-862.

3. **Hochreiter, S., & Schmidhuber, J.** (1997). *Long short-term memory.* Neural Computation, 9(8), 1735-1780.

4. **Box, G. E., et al.** (2015). *Time series analysis: forecasting and control.* John Wiley & Sons.

5. **Efron, B., & Tibshirani, R. J.** (1994). *An introduction to the bootstrap.* CRC Press.

### Documentation Technique

- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [Statsmodels Documentation](https://www.statsmodels.org/)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)

## 📝 License

Ce projet est sous licence MIT - voir le fichier [LICENSE](LICENSE) pour plus de détails.

```
MIT License

Copyright (c) 2026 Aws Ourari, Nairi Najla, Ines Jaziri

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...
```

## 🙏 Remerciements

- Professeurs et encadrants académiques
- Communauté open-source (TensorFlow, Statsmodels)
- Fournisseurs de données boursières
- Tous ceux qui ont contribué à ce projet

## 🐛 Bugs et Suggestions

Si vous trouvez des bugs ou avez des suggestions d'amélioration :

1. **Ouvrir une issue** sur GitHub
2. **Fork** le projet et créer une pull request
3. **Contacter** l'équipe directement

## 📊 Statistiques du Projet

![Lines of Code](https://img.shields.io/badge/Lines%20of%20Code-2000+-blue)
![Commits](https://img.shields.io/badge/Commits-50+-green)
![Contributors](https://img.shields.io/badge/Contributors-3-orange)

---

<div align="center">

**⭐ Si ce projet vous est utile, n'hésitez pas à lui donner une étoile! ⭐**

Made with ❤️ by [Aws Ourari](https://github.com/aws-ourari), [Nairi Najla](https://github.com/nairi-najla), [Ines Jaziri](https://github.com/ines-jaziri)

</div>
