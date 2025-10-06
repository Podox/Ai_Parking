# 🚗 AI Parking Detection System

Un système intelligent de détection de places de parking utilisant l'intelligence artificielle pour surveiller et analyser l'occupation des places de parking en temps réel.

## 📋 Description

Ce projet utilise YOLO (You Only Look Once) pour détecter les véhicules dans un parking et déterminer quelles places sont occupées ou libres. Le système fournit une interface visuelle en temps réel avec un tableau de bord 2D et sauvegarde les statistiques d'utilisation dans une base de données MySQL.

## ✨ Fonctionnalités

- **Détection en temps réel** : Utilise YOLOv8 pour détecter les véhicules
- **Définition des places** : Interface interactive pour définir les zones de parking
- **Surveillance continue** : Suivi de l'occupation des places en temps réel
- **Tableau de bord 2D** : Vue d'ensemble des places avec statut (libre/occupé)
- **Statistiques avancées** : 
  - Nombre de fois qu'une place a été occupée
  - Temps total d'occupation
  - Sauvegarde en CSV et MySQL
- **Interface intuitive** : Affichage visuel avec codes couleur

## 🛠️ Technologies Utilisées

- **Python 3.x**
- **OpenCV** : Traitement d'images et interface graphique
- **YOLOv8** : Détection d'objets en temps réel
- **MySQL** : Base de données pour stockage des statistiques
- **Pandas** : Manipulation des données
- **NumPy** : Calculs numériques
- **Pickle** : Sauvegarde des configurations

## 📦 Installation

### Prérequis

1. **Python 3.7+** installé sur votre système
2. **MySQL** configuré et accessible
3. **Webcam** ou caméra IP pour la détection

### Installation des dépendances

```bash
pip install opencv-python
pip install ultralytics
pip install pandas
pip install numpy
pip install mysql-connector-python
```

### Configuration de la base de données

Créez une base de données MySQL avec la table suivante :

```sql
CREATE DATABASE parking_db;

USE parking_db;

CREATE TABLE parking_usage (
    id INT AUTO_INCREMENT PRIMARY KEY,
    spot_id INT NOT NULL,
    times_occupied INT DEFAULT 0,
    total_time_occupied_seconds DECIMAL(10,2) DEFAULT 0.00,
    is_occupied TINYINT(1) DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## 🚀 Utilisation

### Étape 1 : Définir les places de parking

Exécutez le script de définition des places :

```bash
python parkingspots.py
```

**Instructions :**
- Cliquez avec le bouton gauche pour définir les 4 coins d'une place de parking
- Cliquez avec le bouton droit pour annuler le dernier point
- Appuyez sur `s` pour sauvegarder les places définies
- Appuyez sur `q` pour quitter

### Étape 2 : Lancer la détection

Exécutez le script principal de détection :

```bash
python parkingdetect.py
```

**Fonctionnalités :**
- Détection automatique des véhicules
- Affichage en temps réel du statut des places
- Sauvegarde automatique des statistiques
- Appuyez sur `q` pour quitter

## 📁 Structure du projet

```
ai-parking/
├── parkingdetect.py      # Script principal de détection
├── parkingspots.py       # Script de définition des places
├── parking_spots.pkl     # Fichier de sauvegarde des places
├── yolov8s.pt           # Modèle YOLO pré-entraîné
├── coco.txt             # Classes d'objets COCO
├── park.jpg             # Image de test
├── park1.jpg            # Image de test
├── easy.mp4             # Vidéo de test
├── easy1.mp4            # Vidéo de test
└── README.md            # Documentation
```

## ⚙️ Configuration

### Configuration de la base de données

Modifiez les paramètres de connexion dans `parkingdetect.py` :

```python
db_conn = mysql.connector.connect(
    host="votre_host",
    user="votre_utilisateur",
    password="votre_mot_de_passe",
    database="parking_db"
)
```

### Configuration de la caméra

Pour utiliser une webcam :
```python
cap = cv2.VideoCapture("0")  # 0 pour la webcam par défaut
```

Pour utiliser une vidéo :
```python
cap = cv2.VideoCapture("chemin/vers/votre/video.mp4")
```

## 📊 Sorties

### Fichiers générés

- **`parking_spots.pkl`** : Configuration des places de parking
- **`parking_usage_stats.csv`** : Statistiques d'utilisation en CSV
- **Base de données MySQL** : Statistiques persistantes

### Interface utilisateur

- **Vue principale** : Détection des véhicules avec rectangles de détection
- **Tableau de bord 2D** : Vue d'ensemble des places avec statut
- **Compteur** : Nombre de places libres/total

## 🎯 Utilisation des statistiques

Les statistiques sauvegardées incluent :
- **spot_id** : Identifiant de la place
- **times_occupied** : Nombre de fois occupée
- **total_time_occupied_seconds** : Temps total d'occupation
- **is_occupied** : Statut actuel (0=libre, 1=occupé)

## 🔧 Personnalisation

### Ajustement de la sensibilité

Modifiez le seuil de détection dans le code :
```python
if frame_count % 3 != 0:  # Traite 1 frame sur 3
    continue
```

### Modification des couleurs

Changez les couleurs d'affichage :
```python
color = (0, 0, 255) if currently_occupied else (0, 255, 0)  # Rouge=occupé, Vert=libre
```

## 🐛 Dépannage

### Problèmes courants

1. **Erreur de connexion MySQL** : Vérifiez les paramètres de connexion
2. **Caméra non détectée** : Vérifiez que la webcam est connectée
3. **Modèle YOLO manquant** : Le fichier `yolov8s.pt` sera téléchargé automatiquement

### Logs et débogage

Le système affiche des messages de statut dans la console :
- `✔ Real-time stats saved to CSV and MySQL.`
- `Saved X spots to parking_spots.pkl`


## 👥 Auteurs

- **Amri Badr**
- **Nizar Akka**



**Note** : Ce projet est conçu pour des fins éducatives et de démonstration. Pour une utilisation en production, des améliorations de sécurité et de performance sont recommandées.
