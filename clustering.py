"""
clustering.py — Clustering model fitting, evaluation, and description utilities.

Usage example:
    from clustering import evaluate_k, fit_kmeans, evaluate_k_ts, fit_kshape
    from clustering import describe_clusters, labels_to_dataframe

    # Variation 1 — daily features
    eval_dict = evaluate_k(X, range(2, 11))
    labels, model = fit_kmeans(X, k=5)

    # Variation 2 — multivariate time series
    eval_ts = evaluate_k_ts(X_ts, range(2, 8))
    labels_ts, model_ts = fit_kshape(X_ts, k=5)
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tslearn.clustering import KShape


# ---------------------------------------------------------------------------
# Variation 1 — daily feature k-means
# ---------------------------------------------------------------------------

def evaluate_k(
    X: np.ndarray,
    k_range=range(2, 11),
    n_init: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """Run k-means for each k and return inertia + silhouette scores.

    Parameters
    ----------
    X : np.ndarray, shape (n_days, n_features)
        Normalised feature matrix (no NaNs).
    k_range : iterable of int
        Values of k to evaluate.
    n_init : int
        Number of k-means restarts per k.
    random_state : int

    Returns
    -------
    pd.DataFrame
        Columns: ``k``, ``inertia``, ``silhouette``.
    """
    records = []
    for k in k_range:
        km = KMeans(n_clusters=k, init='k-means++', n_init=n_init, random_state=random_state)
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels) if k > 1 else np.nan
        records.append({'k': k, 'inertia': km.inertia_, 'silhouette': sil})
        print(f'  k={k}: inertia={km.inertia_:.1f}, silhouette={sil:.4f}')
    return pd.DataFrame(records)


def fit_kmeans(
    X: np.ndarray,
    k: int,
    n_init: int = 20,
    random_state: int = 42,
) -> tuple:
    """Fit k-means and return integer cluster labels and the fitted model.

    Parameters
    ----------
    X : np.ndarray, shape (n_days, n_features)
    k : int
        Number of clusters.
    n_init : int
        Number of restarts (more than evaluate_k for stability).
    random_state : int

    Returns
    -------
    labels : np.ndarray, shape (n_days,)
        Integer cluster assignment for each day (0-indexed).
    model : KMeans
        Fitted sklearn KMeans object.
    """
    model = KMeans(n_clusters=k, init='k-means++', n_init=n_init, random_state=random_state)
    labels = model.fit_predict(X)
    sil = silhouette_score(X, labels)
    print(f'KMeans k={k}: inertia={model.inertia_:.1f}, silhouette={sil:.4f}')
    return labels, model


# ---------------------------------------------------------------------------
# Variation 2 — multivariate time-series k-Shape
# ---------------------------------------------------------------------------

def evaluate_k_ts(
    X_ts: np.ndarray,
    k_range=range(2, 8),
    random_state: int = 42,
) -> pd.DataFrame:
    """Run k-Shape for each k and return inertia + silhouette scores.

    Silhouette is computed on the flattened (n_days, 24*3) representation
    with Euclidean distance as a practical proxy (true k-Shape uses SBD).

    Parameters
    ----------
    X_ts : np.ndarray, shape (n_days, 24, 3)
        Multivariate time-series tensor (no NaNs).
    k_range : iterable of int
    random_state : int

    Returns
    -------
    pd.DataFrame
        Columns: ``k``, ``inertia``, ``silhouette``.
    """
    X_flat = X_ts.reshape(len(X_ts), -1)
    records = []
    for k in k_range:
        ks = KShape(n_clusters=k, random_state=random_state)
        labels = ks.fit_predict(X_ts)
        sil = silhouette_score(X_flat, labels) if k > 1 else np.nan
        records.append({'k': k, 'inertia': ks.inertia_, 'silhouette': sil})
        print(f'  k={k}: inertia={ks.inertia_:.4f}, silhouette={sil:.4f}')
    return pd.DataFrame(records)


def fit_kshape(
    X_ts: np.ndarray,
    k: int,
    random_state: int = 42,
) -> tuple:
    """Fit k-Shape on a multivariate time-series tensor.

    Parameters
    ----------
    X_ts : np.ndarray, shape (n_days, 24, 3)
    k : int
        Number of clusters.
    random_state : int

    Returns
    -------
    labels : np.ndarray, shape (n_days,)
    model : KShape
        Fitted tslearn KShape object.  Centroids are at ``model.cluster_centers_``
        with shape ``(k, 24, 3)``.
    """
    model = KShape(n_clusters=k, random_state=random_state)
    labels = model.fit_predict(X_ts)
    X_flat = X_ts.reshape(len(X_ts), -1)
    sil = silhouette_score(X_flat, labels)
    print(f'KShape k={k}: inertia={model.inertia_:.4f}, silhouette={sil:.4f}')
    return labels, model


# ---------------------------------------------------------------------------
# Post-clustering analysis
# ---------------------------------------------------------------------------

def _raw_name(col: str) -> str:
    """Strip normalisation suffixes (_z, _sc) to recover the original column name."""
    for suffix in ('_z', '_sc'):
        if col.endswith(suffix):
            return col[: -len(suffix)]
    return col


def describe_clusters(
    meta_df: pd.DataFrame,
    labels: np.ndarray,
    daily_df: pd.DataFrame,
    feature_cols: list,
) -> pd.DataFrame:
    """Compute a per-cluster summary table for interpretation.

    ``feature_cols`` may contain normalised names with ``_z`` / ``_sc`` suffixes
    (as returned by ``build_daily_features``); the function strips those suffixes
    automatically to look up raw values in ``daily_df``.

    Parameters
    ----------
    meta_df : pd.DataFrame
        Metadata aligned with ``labels`` — must contain ``id`` and ``date``.
        Optionally ``disease_type``, ``sex``, ``age``.
    labels : np.ndarray, shape (n_days,)
        Integer cluster assignments from ``fit_kmeans`` or ``fit_kshape``.
    daily_df : pd.DataFrame
        Original (un-normalised) ``daily`` DataFrame for readable centroid values.
        Joined on ``id`` + ``date`` to retrieve raw feature values.
    feature_cols : list[str]
        Feature columns used in clustering (may include ``_z``/``_sc`` suffixes).

    Returns
    -------
    pd.DataFrame
        One row per cluster with columns:
        ``n_days``, ``pct_total``,
        one mean column per raw feature (``mean_<original_name>``),
        ``pct_early``, ``pct_fast``, ``pct_late`` (disease stage breakdown).
    """
    tagged = meta_df.copy()
    tagged['cluster'] = labels

    # Map suffixed → raw names; only keep those present in daily_df
    raw_col_map = {col: _raw_name(col) for col in feature_cols}
    raw_cols    = [rc for rc in raw_col_map.values() if rc in daily_df.columns]
    raw = daily_df[['id', 'date'] + raw_cols].copy()
    tagged = tagged.merge(raw, on=['id', 'date'], how='left')

    records = []
    n_total = len(tagged)
    for cl in sorted(tagged['cluster'].unique()):
        sub = tagged[tagged['cluster'] == cl]
        row = {'cluster': cl, 'n_days': len(sub), 'pct_total': len(sub) / n_total * 100}

        for raw_col in raw_cols:
            row[f'mean_{raw_col}'] = sub[raw_col].mean()

        if 'disease_type' in sub.columns:
            vc = sub['disease_type'].value_counts(normalize=True) * 100
            row['pct_early'] = vc.get('Early Disease Stage', 0.0)
            row['pct_fast']  = vc.get('Fast Disease Progression', 0.0)
            row['pct_late']  = vc.get('Late Disease Stage', 0.0)

        records.append(row)

    return pd.DataFrame(records).set_index('cluster')


def feature_importance_anova(
    X: np.ndarray,
    labels: np.ndarray,
    feature_cols: list,
) -> pd.DataFrame:
    """Rank features by how much they differ across clusters (one-way ANOVA).

    A high F-statistic means between-cluster variance is large relative to
    within-cluster variance — i.e. that feature drives cluster separation.

    Parameters
    ----------
    X : np.ndarray, shape (n_days, n_features)
        Normalised feature matrix (same as passed to ``fit_kmeans``).
    labels : np.ndarray, shape (n_days,)
        Cluster assignments.
    feature_cols : list[str]
        Feature names for each column of X.

    Returns
    -------
    pd.DataFrame
        Columns: ``feature``, ``raw_feature``, ``F_stat``, ``p_value``,
        sorted descending by ``F_stat``.
    """
    from scipy.stats import f_oneway

    unique_clusters = np.unique(labels)
    records = []
    for j, col in enumerate(feature_cols):
        groups = [X[labels == k, j] for k in unique_clusters]
        f_stat, p_val = f_oneway(*groups)
        records.append({
            'feature':     col,
            'raw_feature': _raw_name(col),
            'F_stat':      f_stat,
            'p_value':     p_val,
        })
    return pd.DataFrame(records).sort_values('F_stat', ascending=False).reset_index(drop=True)


def labels_to_dataframe(
    meta_df: pd.DataFrame,
    labels: np.ndarray,
    label_col: str = 'cluster',
    date_col: str = 'date',
) -> pd.DataFrame:
    """Wrap cluster labels as a tidy DataFrame for merging back into ``daily``.

    Parameters
    ----------
    meta_df : pd.DataFrame
        Metadata aligned with ``labels``.  Must contain ``id`` and either
        ``date`` or ``shifted_date``.
    labels : np.ndarray, shape (n_days,)
    label_col : str
        Name for the cluster column in the output.
    date_col : str
        Name of the date column in ``meta_df`` to carry through.

    Returns
    -------
    pd.DataFrame
        Columns: ``id``, ``date`` (renamed from ``date_col``), ``label_col``.
    """
    out = meta_df[['id', date_col]].copy().reset_index(drop=True)
    out = out.rename(columns={date_col: 'date'})
    out[label_col] = labels
    return out
