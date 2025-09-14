# Rendu Examen - Drift Monitoring
Documentation d'accompagnement pour l'examen Drift Monitoring (basé sur evidently)

## 1. Réponses aux questions des étapes 4 à 6

### Etape 4 :

### Etape 5 :

### Etape 6 :

## 2. Commande centralisée d'appel de la chaine de traitement

### Setup préalable de l'environnement virtuel

Utilisation d'un pyproject.toml équivalent au requirements.txt demandé

```bash
# A executer dans le repository cloné
uv sync
source .venv/bin/activate
```

### Execution de la chaîne de traitement (et visualisation via evidently)

```bash
### Appel de traitement
# Les rapports HTML/JSON sont générés dans `reports/exam_drift_monitoring/`.
python build_reports.py --outdir reports --project exam_drift_monitoring
# Visualisation dans evidently UI
evidently ui --workspace ./exam_drift_monitoring_workspace
```

## 3. [Optionnel] Rappel des vérification d'usage de la VM (avec OS Ubuntu)

Non demandé mais par acquis de conscience si UV n'est encore présent...

### Mise à jour globale des paquets et rattrapage éventuel des installations manquantes

```bash
# Mise à jour de la liste des paquets
sudo apt update
# Récupère et corrige d'éventuels paquets manquants
sudo apt install --fix-missing
```

### Mise à jour python3

```bash
# (Ré)installation/verif de python3 (par sureté)
sudo apt install -y python3
```

### Mise à jour pip3 et pipx

```bash
# (Ré)installation/verif de pip3 et pipx (par sureté)
sudo apt-get install -y python3-pip
sudo apt install -y pipx
pipx ensurepath
source ~/.bashrc
```

### Installation UV (gestionnaire environnement virtuel)

```bash
# Lance l’installation de uv
pipx install uv
```

### Check final des composants nécessaires (ils doivent être tous présents)

```bash
# l'ensemble des composants de base est présent
python3 --version
pip3 --version
pipx --version
uv --version
```


