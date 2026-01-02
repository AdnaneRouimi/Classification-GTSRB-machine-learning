
# Classification des panneaux de signalisation routière (GTSRB)

## Description du projet

Ce projet vise à développer un système de vision par ordinateur capable de classifier automatiquement les panneaux de signalisation routière à partir d’images.  
Il repose sur des techniques de machine learning et de deep learning appliquées au dataset GTSRB (German Traffic Sign Recognition Benchmark).

Le système permet :
- la classification d’images statiques,
- l’inférence en temps réel via webcam,
- l’utilisation du modèle à travers une application web développée avec Streamlit.

---

## Objectifs pédagogiques

- Mettre en œuvre un apprentissage supervisé sur un dataset d’images réelles  
- Comprendre le pipeline complet de classification d’images  
- Utiliser un réseau de neurones convolutif (CNN)  
- Tester un modèle sur images statiques et en temps réel  
- Déployer un modèle via une application simple  

---

## Approche technique

- Dataset : GTSRB (43 classes de panneaux)
- Prétraitement :
  - Organisation des données en ensembles train et validation
  - Renommage des classes en français
- Modèle :
  - YOLOv8 en mode classification
- Entraînement :
  - Apprentissage supervisé avec accélération GPU (CUDA)
- Inférence :
  - Image statique
  - Webcam
  - Application web Streamlit

---

## Structure du projet
 
├── cls_dataset/ # Dataset de classification (train / val)  
├── data/ # Dataset original GTSRB  
├── download/ # Scripts ou fichiers de téléchargement  
├── runs/classify/ # Résultats d’entraînement (poids, logs)  
├── yolo_dataset/ # Tentative initiale de détection (non utilisée)  
├── main.ipynb # Notebook principal  
├── streamlit_app.py # Application web Streamlit  
├── yolov8n-cls.pt # Modèle YOLOv8 classification pré-entraîné  
├── yolov8n.pt # Modèle YOLOv8 générique  
├── README.md # Documentation du projet  
└── .gitignore.txt # Fichiers ignorés par Git

---

## Prérequis

- Python 3.9 ou supérieur
- Environnement virtuel recommandé
- GPU compatible CUDA (optionnel mais recommandé)

---

## Installation des dépendances

Créer et activer un environnement virtuel, puis installer les dépendances suivantes :

```bash
pip install ultralytics
pip install torch torchvision
pip install opencv-python
pip install streamlit
pip install pillow
pip install matplotlib
----------

## Entraînement du modèle

L’entraînement du modèle est réalisé depuis le notebook principal :

`main.ipynb` 

Ce notebook inclut :

-   la préparation du dataset de classification,
    
-   l’entraînement du modèle YOLOv8,
    
-   la validation et les tests sur images.
    

----------

## Lancement de l’application web

Pour lancer l’application Streamlit permettant de tester le modèle :

`streamlit run streamlit_app.py` 

L’application est accessible à l’adresse suivante :

`http://localhost:8501` 

Fonctionnalités disponibles :

-   chargement d’une image par l’utilisateur,
    
-   utilisation de la webcam,
    
-   sélection aléatoire d’une image du dataset,
    
-   affichage de la classe prédite et du score de confiance.
    

----------

## Résultats

Le modèle est capable de reconnaître correctement la majorité des panneaux du dataset GTSRB.  
Les prédictions sont accompagnées d’un score de confiance et peuvent être réalisées en temps réel.

----------

## Perspectives

-   Ajout de la détection des panneaux dans des scènes complexes
    
-   Amélioration des performances du modèle
    
-   Déploiement sur systèmes embarqués ou plateformes cloud
    
-   Extension vers des scénarios de conduite réels
    

----------
