# Rendu Examen - Drift Monitoring
Documentation d'accompagnement pour l'examen Drift Monitoring (basé sur evidently)

## 1. Réponses aux questions après les étapes 4 à 6

### Après l'étape 4, expliquez ce qui a changé au cours des semaines 1, 2 et 3 :

En analysant 03_model_drift_feb_week{1,2,3}.html (ou l'équivalent sur evidently UI) on peut constater une évolution à la hausse à chaque nouvelle semaine

- de l'erreur moyenne (ME = -6,27 / -9,75 / -22,65) qui s'accentue dans le négatif montrant que le modèle sous estime de plus en plus la prédiction réelle au fil des 3 semaines
- de l'erreur absolue moyenne (MAE = 13,53 / 15,21 / 24,76) qui s'accentue dans le positif, soulignant que la distance à la réalité se creuse au fil des 3 semaines
- du l'erreur absolue moyenne en pourcentage (MAPE = 41,58% / 32,25% / 40,43%), qui reste importante par rapport aux données d'entrainement (16,2%)

Cela démontre une dégradation des performances du modèles qui s'accentue avec le temps.

### Après l'étape 5, expliquez ce qui semble être la cause première de la dérive (uniquement à l'aide de données) :

La distribution des valeurs de la variable cible sur la 3ème semaine de février est répartie globalement sur des valeurs plus hautes que sur la période de référence de janvier (notamment au dela de 100 et jusqu'à 305 alors que la référence s'arrête à 218 )

La hausse du trafic sur la semaine 3 de février comparé a tout janvier explique donc très certainement la dégradation des résultats (confirmation de l'observation de ME qui devient de plus en plus intensément négatif).

### Après l'étape 6, expliquez quelle stratégie appliquer :

Les variables explicatives à distribution stable (p-values élevées) sont:
- les données périodiques : hr (heure du jour) et weekday (numéro du jour de la semaine)

Les variables explicatives à distribution en dérive (p-value < 0.05) sont:
- la donnée périodique : mnth (numéro du mois passant de 1 pour janvier à 2 pour février : donc un changement structurel de numéro de mois prévisible)
- les données de conditions météos : temp (température réelle), atemp (température ressentie), hum (humidité), windspeed (force du vent) qui dénote un vrai drift prévisible des données climatiques entre janvier et cette semaine de février (on constate par exemple une distribution qui shifte vers les valeurs plus hautes)

La cause potentielle de la dérive est donc que le modèle n'a pas encore bien appris de l'influence des variations météos sur l'intensité du trafic et il faut mettre en place une stratégie de rafraichissement sur fenetre mobile pour prendre en compte ce point. Par exemple un ré-entrainement à fréquence régulière couplé à un réentrainement suite à observation d'une p-value sur données météos inférieure au seuil de 0.05 (vu qu'un changement climatique à un effet statistiquement notable sur la valeur de comptage).

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
python build_reports.py --outdir reports --project exam_drift_monitoring --ui-local --ui-workspace .exam_evidently_workspace
# Visualisation dans evidently UI
evidently ui --workspace ./.exam_evidently_workspace
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


