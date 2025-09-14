
## Contenu généré

* `01_validation.*` : Validation du modèle (train vs test sur Jan 1–28)
* `02_prod_january.*` : Perf. du modèle de production (entraîné sur Jan complet)
* `03_week{1,2,3}_regression.*` : Suivi hebdo (février)
* `04_target_drift_<worstweek>.*` : Dérive de la **cible** sur la semaine la plus défavorable
* `05_data_drift_week3_numeric.*` : Dérive **des données** (numérique uniquement) sur la dernière semaine
* `00_summary.json` : MAE par semaine et semaine la plus défavorable

## Réponses attendues

### Après l'étape 4 — Qu’est-ce qui a changé entre les semaines 1, 2 et 3 ?

> Appuyez-vous sur `03_week*.html` + `00_summary.json` (MAE par semaine). Décrivez l’évolution des erreurs, des distributions de features clés et des résidus.

### Après l'étape 5 — Quelle semble être la cause première de la dérive (côté **cible** uniquement) ?

> Appuyez-vous sur `04_target_drift_<worstweek>.html` (tests, p-values, graphiques de distribution de la cible).

### Après l'étape 6 — Quelle stratégie appliquer ?

> Appuyez-vous sur `05_data_drift_week3_numeric.html` pour relier la dérive **données** aux métriques. Proposez :
>
> * Recalibrage (retrain partiel avec fenêtre glissante),
> * Mise à jour des features (ex. agrégations temporelles),
> * Ajustement des seuils/alertes.

## Notes

* Les rapports sont autonomes (HTML). Ils peuvent être importés/managés dans un même projet Evidently UI si vous utilisez un Workspace.

---

## ✅ Check-list d’exécution

1. Créer et activer un venv (recommandé)
2. `pip install -r requirements.txt`
3. `python build_reports.py --outdir reports --project bike-jan-feb-2011`
4. Ouvrir les HTML générés et compléter le README.
