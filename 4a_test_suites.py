"""
Evidently – suites de tests pour le projet "test_suites2".

Ce module propose :
- Une suite **sans cible** via `NoTargetPerformanceTestPreset` (batches de données).
- Une suite **custom** avec métriques (précision, rappel, accuracy) **si et seulement si**
une cible est disponible.
- Un rapport de dérive **par colonne(s)** via `ColumnDriftMetric`.

Organisation des artefacts : `./artifacts/test_suites2/`

Notes typage / Pylance :
- Certaines signatures Evidently (Pydantic v2) peuvent déclencher des warnings.
- Voir le cast sur la liste de tests si Pylance râle.

Pré-requis :
    pip install evidently pandas numpy

Auteur : vous 😉
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Tuple, cast

import numpy as np
import pandas as pd

# Evidently – imports principaux
from evidently.report import Report
from evidently.test_preset.no_target_performance import NoTargetPerformanceTestPreset
from evidently.test_suite import TestSuite

# Métriques / tests spécifiques (certaines nécessitent une cible/pred)
from evidently.tests import (
    TestAccuracyScore,
    TestPrecisionScore,
    TestRecallScore,
    TestShareOfMissingValues,
)
from evidently.metrics import ColumnDriftMetric


# -----------------------------------------------------------------------------
# Configuration projet
# -----------------------------------------------------------------------------
PROJECT = "test_suites2"
ARTIFACTS_DIR = Path("artifacts") / PROJECT
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Utilitaires
# -----------------------------------------------------------------------------
@dataclass
class BatchWindow:
    start: int
    end: int


def split_into_batch_windows(n_rows: int, batch_size: int) -> List[BatchWindow]:
    """Retourne des fenêtres [start:end] sans chevauchement.
    La dernière fenêtre peut être plus petite si n_rows n'est pas multiple.
    """
    windows: List[BatchWindow] = []
    for start in range(0, n_rows, batch_size):
        end = min(n_rows, start + batch_size)
        windows.append(BatchWindow(start, end))
    return windows


# -----------------------------------------------------------------------------
# Suites SANS cible
# -----------------------------------------------------------------------------

def run_no_target_performance_suites(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    batch_size: int = 10_000,
    prefix: str = "no_target_perf",
) -> List[Tuple[BatchWindow, Path]]:
    """Exécute la `NoTargetPerformanceTestPreset` sur des batches successifs de `current_df`.

    Args:
        reference_df: données de référence (mêmes features que current_df)
        current_df: données de prod (ou simulation) SANS variable cible
        batch_size: taille d'un batch
        prefix: préfixe des fichiers HTML

    Returns:
        Liste de tuples (fenêtre, chemin_html)
    """
    outputs: List[Tuple[BatchWindow, Path]] = []
    windows = split_into_batch_windows(len(current_df), batch_size)

    for idx, win in enumerate(windows):
        batch = current_df.iloc[win.start:win.end]

        # Construction de la suite (cast pour Pylance si besoin)
        tests_list = cast(List[Any], [NoTargetPerformanceTestPreset()])
        suite = TestSuite(tests=tests_list)
        suite.run(reference_data=reference_df, current_data=batch)

        html_path = ARTIFACTS_DIR / f"{prefix}_batch{idx:03d}_{win.start}-{win.end}.html"
        suite.save_html(str(html_path))
        outputs.append((win, html_path))

    return outputs


# -----------------------------------------------------------------------------
# Suites AVEC métriques custom (néc. d'une cible et, le plus souvent, de prédictions)
# -----------------------------------------------------------------------------

def has_columns(df: pd.DataFrame, required: Sequence[str]) -> bool:
    return all(col in df.columns for col in required)


def run_custom_metrics_suite(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    target_col: str = "target",
    pred_col: str = "prediction",
    prefix: str = "custom_metrics",
    thresholds: Optional[dict] = None,
) -> Optional[Path]:
    """Exécute une suite de tests custom si la cible (et idéalement la prédiction) existent.

    Par défaut, applique les seuils suivants si non fournis :
        share_missing == 0, precision > 0.5, recall > 0.3, accuracy >= 0.75.

    Returns:
        Chemin du HTML, ou None si colonnes manquantes.
    """
    thresholds = thresholds or {
        "share_missing_eq": 0.0,
        "precision_gt": 0.5,
        "recall_gt": 0.3,
        "accuracy_gte": 0.75,
    }

    # Vérifications de présence des colonnes nécessaires
    need_ref = has_columns(reference_df, [target_col, pred_col])
    need_cur = has_columns(current_df, [target_col, pred_col])

    if not (need_ref and need_cur):
        # Pas de cible/prédictions -> on ne lance pas cette suite.
        return None

    tests = [
        TestShareOfMissingValues(eq=thresholds["share_missing_eq"]),  # type: ignore
        TestPrecisionScore(gt=thresholds["precision_gt"]),  # type: ignore
        TestRecallScore(gt=thresholds["recall_gt"]),  # type: ignore
        TestAccuracyScore(gte=thresholds["accuracy_gte"]),  # type: ignore
    ]

    # Astuce lisibilité : expliciter le seuil dans le titre (incompatible evidently 0.6.7)
    # tests[1].display_name = f"Precision > {thresholds['precision_gt']:.2f}"
    # tests[2].display_name = f"Recall > {thresholds['recall_gt']:.2f}"
    # tests[3].display_name = f"Accuracy ≥ {thresholds['accuracy_gte']:.2f}"

    suite = TestSuite(tests=cast(List[Any], tests))
    suite.run(reference_data=reference_df, current_data=current_df)

    html_path = ARTIFACTS_DIR / f"{prefix}.html"
    suite.save_html(str(html_path))
    return html_path


# -----------------------------------------------------------------------------
# Rapports de dérive par colonne(s)
# -----------------------------------------------------------------------------

def build_column_drift_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    columns: Iterable[str],
    use_psi_for: Optional[Iterable[str]] = None,
    prefix: str = "column_drift",
) -> Path:
    """Construit un `Report` Evidently avec des `ColumnDriftMetric` pour les colonnes demandées.

    Args:
        reference_df: données de référence
        current_df: données actuelles
        columns: liste des colonnes à évaluer
        use_psi_for: sous-ensemble de `columns` pour lequel on force `stattest='psi'`
        prefix: préfixe du fichier HTML

    Returns:
        Chemin du rapport HTML généré
    """
    use_psi_for = set(use_psi_for or [])

    metrics: List[ColumnDriftMetric] = []
    for col in columns:
        if col in use_psi_for:
            metrics.append(ColumnDriftMetric(column_name=col, stattest="psi"))
        else:
            metrics.append(ColumnDriftMetric(column_name=col))

    report = Report(metrics=metrics)  # type: ignore
    report.run(reference_data=reference_df, current_data=current_df)

    html_path = ARTIFACTS_DIR / f"{prefix}.html"
    report.save_html(str(html_path))
    return html_path


# -----------------------------------------------------------------------------
# Exemple d'utilisation (à adapter à votre workspace)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Exemples synthétiques pour montrer l'orchestration.
    # Remplacez par vos vrais DataFrames pré-traités :
    rng = np.random.default_rng(42)

    # Données de référence (sans cible pour la 1ère partie)
    reference = pd.DataFrame({
        "ArrDelay": rng.normal(loc=2.0, scale=5.0, size=30_000),
        "DepDelay": rng.normal(loc=1.0, scale=3.0, size=30_000),
        "Distance": rng.integers(100, 3000, size=30_000),
    })

    # Données actuelles simulées (légère dérive)
    current = pd.DataFrame({
        "ArrDelay": rng.normal(loc=3.0, scale=5.5, size=25_000),
        "DepDelay": rng.normal(loc=1.5, scale=3.5, size=25_000),
        "Distance": rng.integers(100, 3000, size=25_000),
    })

    # 1) Suite SANS cible (par batches)
    print("➡️  Lancement NoTargetPerformanceTestPreset par batches…")
    outputs = run_no_target_performance_suites(
        reference_df=reference,
        current_df=current,
        batch_size=10_000,
        prefix="no_target_perf_demo",
    )
    for win, path in outputs:
        print(f"Batch {win.start}:{win.end} → {path}")

    # 2) (Optionnel) Ajouter une cible/prédiction pour montrer la suite custom
    reference_with_y = reference.copy()
    current_with_y = current.copy()

    # Cible binaire simulée + prédiction (exemple). Remplacez par vos colonnes réelles.
    reference_with_y["target"] = (reference["ArrDelay"] > 3).astype(int)
    current_with_y["target"] = (current["ArrDelay"] > 3).astype(int)
    reference_with_y["prediction"] = (
        reference["ArrDelay"] + rng.normal(0, 1, len(reference)) > 3
    ).astype(int)
    current_with_y["prediction"] = (
        current["ArrDelay"] + rng.normal(0, 1, len(current)) > 3
    ).astype(int)

    print("➡️  Lancement suite custom (avec cible/prédiction)…")
    html_custom = run_custom_metrics_suite(
        reference_df=reference_with_y,
        current_df=current_with_y,
        target_col="target",
        pred_col="prediction",
        thresholds={
            "share_missing_eq": 0.0,
            "precision_gt": 0.50,
            "recall_gt": 0.30,
            "accuracy_gte": 0.75,
        },
    )
    if html_custom:
        print(f"Suite custom → {html_custom}")
    else:
        print("Suite custom non lancée (colonnes manquantes).")

    # 3) Rapport de dérive par colonne(s)
    print("➡️  Rapport de dérive par colonne(s)…")
    drift_html = build_column_drift_report(
        reference_df=reference,
        current_df=current,
        columns=["ArrDelay", "DepDelay"],
        use_psi_for=["ArrDelay"],
        prefix="drift_arrdep_demo",
    )
    print(f"Report column drift → {drift_html}")
