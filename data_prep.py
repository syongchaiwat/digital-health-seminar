"""
data_prep.py — Data preparation utilities for the digital health seminar project.

Usage example:
    from data_prep import get_complete_patient_days, load_weather, load_calendar
    from data_prep import load_covid, load_pollen, load_air_quality

    df_filtered  = get_complete_patient_days(df, shift_hour=6, complete=True)
    weather      = load_weather('data/external/weather/ogd-smn_klo_h_historical_2020-2029.csv')
    calendar     = load_calendar('data/external/holidays/zurich_calendar_2021.csv')
    covid        = load_covid('data/external/covid/ch_stringency_2021.csv')
    pollen       = load_pollen('data/external/pollen/ogd-pollen_pzh_d_historical.csv')
    air_quality  = load_air_quality('data/external/airquality/nabel_zue_2021.csv')

Column name conventions
-----------------------
* Date columns are always named ``date`` (dtype datetime64[ns], midnight-normalised).
* Datetime columns are always named ``datetime`` (dtype datetime64[ns], sub-day precision).
* All other column names use snake_case with explicit units where useful
  (e.g. ``temp_max``, ``precip_total``, ``pm25_mean``).
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def get_complete_patient_days(
    df: pd.DataFrame,
    shift_hour: int = 6,
    complete: bool = True,
) -> pd.DataFrame:
    """Return hourly sensor rows annotated with a shifted date, optionally
    filtered to only complete patient-days (24 valid hourly entries).

    Parameters
    ----------
    df : pd.DataFrame
        Hourly sensor DataFrame with at minimum columns:
        ``id`` (patient), ``time`` (datetime64), ``steps``, ``sleep``,
        ``heartrate``.
    shift_hour : int
        Number of hours to shift the timestamp forward before assigning a
        calendar date.  A value of 6 means that an entry recorded at
        2021-01-02 03:00 is attributed to 2021-01-01 (shifted date), which
        keeps a night's sleep on one logical date instead of splitting it at
        midnight.
    complete : bool
        If True, keep only patient-days that have exactly 24 hourly entries
        in the shifted window (i.e., fully observed days).  If False, all
        rows are returned with the shifted date column attached.

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` with two additional columns:

        * ``shifted_datetime`` — original timestamp shifted forward by
          ``shift_hour`` hours.
        * ``shifted_date``     — calendar date derived from
          ``shifted_datetime`` (midnight-normalised).

        When ``complete=True``, only rows belonging to patient-days with
        exactly 24 hourly entries are included.  The column
        ``n_complete_days`` (count of complete days per patient) is also
        added in that case.
    """
    df = df.copy()

    # Build shifted timestamp and date
    df['shifted_datetime'] = df['time'] + pd.Timedelta(hours=shift_hour)
    df['shifted_date']     = df['shifted_datetime'].dt.floor('D')

    if not complete:
        return df

    # Count entries per (patient, shifted_date); keep only days with 24
    entries_per_day = df.groupby(['id', 'shifted_date']).size().rename('n_entries')
    complete_days   = entries_per_day[entries_per_day == 24].reset_index()[['id', 'shifted_date']]

    df = df.merge(complete_days, on=['id', 'shifted_date'], how='inner')
    df['n_complete_days'] = df.groupby('id')['shifted_date'].transform('nunique')

    return df


# ---------------------------------------------------------------------------
# Daily aggregation
# ---------------------------------------------------------------------------

def _per_day_features(g: pd.DataFrame) -> pd.Series:
    """Compute physiological summary features for one patient-day group."""
    hr        = g['heartrate']
    steps     = g['steps']
    sleep     = g['sleep']
    hour      = g['hour']

    # Heart-rate windows (original clock hour, not shifted)
    resting   = hr[sleep > 30]          # HR during hours with >30 min sleep (true sleep HR)
    active    = g[steps > 0]
    day_mask  = hour.between(8, 20)

    day_hr      = hr[day_mask].mean()
    resting_hr  = resting.mean()

    # Cardiac efficiency: mean HR per 100 steps during active hours
    hr_per_100 = np.nan
    if len(active) >= 2 and active['steps'].mean() > 0:
        hr_per_100 = active['heartrate'].mean() / active['steps'].mean() * 100

    # Step distribution features
    total_steps  = steps.sum()
    step_peak    = steps.max()
    peak_hour    = hour[steps.idxmax()] if total_steps > 0 else np.nan

    # Shannon entropy of hourly step distribution (higher = more evenly spread)
    step_entropy = np.nan
    if total_steps > 0:
        probs        = steps / total_steps
        probs        = probs[probs > 0]
        step_entropy = float(-(probs * np.log(probs)).sum())

    # Sleep fragmentation — two complementary metrics
    sleep_binary  = sleep.gt(0).astype(int)

    # (1) Transition count: number of sleep↔wake switches across the full day
    fragmentation_count = int(sleep_binary.diff().abs().sum())

    # (2) Fragmentation in minutes: wakefulness *within* the sleep episode.
    #     Steps: find the first and last hour with sleep > 0 (episode boundaries),
    #     then sum (60 - sleep_minutes) for every hour inside that window.
    #     Partial hours at onset/offset are intentionally included — they reflect
    #     real lost sleep — but hours outside the episode are excluded.
    sleep_sorted  = sleep.sort_index()   # ensure hours are in order
    nonzero_idx   = sleep_sorted[sleep_sorted > 0].index
    if len(nonzero_idx) > 0:
        episode       = sleep_sorted.loc[nonzero_idx[0]:nonzero_idx[-1]]
        frag_min      = int(np.clip(60 - episode, 0, None).sum())
    else:
        frag_min      = 0

    # Sleep timing within the night window (20:00–06:00 by clock hour)
    night_sleep_mask = (hour >= 20) | (hour <= 6)
    sleep_hours_idx  = hour[night_sleep_mask & (sleep > 0)]
    sleep_onset      = float(sleep_hours_idx.min()) if len(sleep_hours_idx) > 0 else np.nan
    sleep_end        = float(sleep_hours_idx.max()) if len(sleep_hours_idx) > 0 else np.nan
    with_sleep_hour  = int(sleep.gt(0).sum())

    return pd.Series({
        # Heart rate
        'resting_hr':          float(resting_hr),
        'day_hr':              float(day_hr),
        'hr_day_night_delta':  float(day_hr - resting_hr)
                               if pd.notna(day_hr) and pd.notna(resting_hr) else np.nan,
        'hr_var':              float(hr.std()),
        'max_hr':              float(hr.max()),
        'min_hr':              float(hr.min()),
        'hr_per_100steps':     hr_per_100,
        # Activity
        'active_hours':        int(steps.gt(0).sum()),
        'sedentary_hours':     int(steps.eq(0).sum()),
        'step_peak':           float(step_peak),
        'peak_steps_hour':     float(peak_hour),
        'step_entropy':        step_entropy,
        # Sleep
        'sleep_fragmentation':     fragmentation_count,
        'sleep_fragmentation_min': frag_min,
        'sleep_onset_hour':        sleep_onset,
        'sleep_end_hour':      sleep_end,
        'with_sleep_hour':       with_sleep_hour,
    })


def aggregate_to_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate hourly patient data to daily-level features.

    Designed to receive the output of ``get_complete_patient_days`` with
    ``complete=True``.  Groups by ``(id, shifted_date)`` and any clinical
    columns found in the DataFrame (``disease_type``, ``sex``, ``age``).

    The ``shifted_date`` column becomes ``date`` in the output, so all days
    are aligned to the logical day defined by ``shift_hour`` (e.g. 18:00
    previous evening → 17:59 current evening for ``shift_hour=6``).

    Parameters
    ----------
    df : pd.DataFrame
        Hourly sensor DataFrame produced by ``get_complete_patient_days``.
        Required columns: ``id``, ``time``, ``shifted_date``, ``steps``,
        ``sleep``, ``heartrate``.  Optional clinical columns
        ``disease_type``, ``sex``, ``age`` are carried through if present.

    Returns
    -------
    pd.DataFrame
        One row per ``(id, date)`` with columns:

        **Identity / metadata**
        ``id``, ``date``, ``n_complete_days``,
        ``disease_type`` / ``sex`` / ``age`` (if present in input)

        **Steps**
        ``daily_steps``, ``active_hours``, ``sedentary_hours``,
        ``step_peak``, ``peak_steps_hour``, ``step_entropy``

        **Sleep**
        ``sum_sleep_minute``, ``with_sleep_hour``,
        ``sleep_fragmentation`` (transition count),
        ``sleep_fragmentation_min`` (wakefulness minutes inside sleep episode),
        ``sleep_onset_hour``, ``sleep_end_hour``

        **Heart rate**
        ``mean_hr``, ``max_hr``, ``min_hr``, ``hr_var``,
        ``resting_hr`` (mean HR during hours with >30 min sleep), ``day_hr`` (08:00–20:00),
        ``hr_day_night_delta`` (day_hr − resting_hr), ``hr_per_100steps``
    """
    df = df.copy()

    # Ensure hour is available for within-day feature computation
    if 'hour' not in df.columns:
        df['hour'] = df['time'].dt.hour

    # Determine groupby keys
    clinical_cols = [c for c in ['disease_type', 'sex', 'age'] if c in df.columns]
    group_keys    = ['id', 'shifted_date'] + clinical_cols

    # ── Core aggregations ────────────────────────────────────────────────────
    core = (
        df.groupby(group_keys, sort=False)
        .agg(
            n_complete_days  = ('n_complete_days', 'first'),
            daily_steps      = ('steps',           'sum'),
            sum_sleep_minute = ('sleep',           'sum'),
            mean_hr          = ('heartrate',        'mean'),
        )
        .reset_index()
    )

    # ── Per-day physiological features ───────────────────────────────────────
    print('Computing per-day features...')
    day_feat = (
        df.groupby(['id', 'shifted_date'], sort=False)
        .apply(_per_day_features, include_groups=False)
        .reset_index()
    )

    daily = core.merge(day_feat, on=['id', 'shifted_date'], how='left')
    daily = daily.rename(columns={'shifted_date': 'date'})

    # Canonical column order
    id_cols   = ['id', 'date', 'n_complete_days'] + clinical_cols
    step_cols = ['daily_steps', 'active_hours', 'sedentary_hours',
                 'step_peak', 'peak_steps_hour', 'step_entropy']
    sleep_cols = ['sum_sleep_minute', 'with_sleep_hour',
                  'sleep_fragmentation', 'sleep_fragmentation_min',
                  'sleep_onset_hour', 'sleep_end_hour']
    hr_cols   = ['mean_hr', 'max_hr', 'min_hr', 'hr_var',
                 'resting_hr', 'day_hr',
                 'hr_day_night_delta', 'hr_per_100steps']

    ordered = [c for c in id_cols + step_cols + sleep_cols + hr_cols
               if c in daily.columns]
    return daily[ordered]


# ---------------------------------------------------------------------------
# Clustering feature preparation
# ---------------------------------------------------------------------------

# HR features normalised within-patient (individual baseline differs).
_HR_FEATURES = ['resting_hr', 'hr_var', 'hr_day_night_delta', 'hr_per_100steps']

# Step / sleep features normalised globally (preserve between-patient magnitude).
_STEP_SLEEP_FEATURES = [
    'daily_steps', 'active_hours', 'step_peak', 'step_entropy',
    'sum_sleep_minute', 'sleep_fragmentation_min', 'with_sleep_hour',
]

# External features added in round 2 (all global).
_EXTERNAL_FEATURES = [
    'temp_max', 'sunshine_total', 'pollen_total', 'is_weekend', 'stringency_index',
]


def build_daily_features(
    daily_df: pd.DataFrame,
    include_external: bool = False,
) -> tuple:
    """Build a normalised feature matrix for daily-level clustering (Variation 1).

    Applies a hybrid normalisation strategy:

    * **HR features** — within-patient z-score so individual HR baselines do not
      dominate cluster assignment.
    * **Step / sleep features** — global ``StandardScaler`` so between-patient
      magnitude differences (e.g. late-stage patients taking fewer steps) are
      preserved and can drive cluster structure.
    * **External features** (optional) — global ``StandardScaler``.

    Parameters
    ----------
    daily_df : pd.DataFrame
        Output of ``aggregate_to_daily()``, optionally with external columns
        already merged in (``temp_max``, ``sunshine_total``, ``pollen_total``,
        ``is_weekend``, ``stringency_index``).
    include_external : bool
        If True, append the five external features to the feature matrix.
        Rows missing any external value are dropped.

    Returns
    -------
    X : np.ndarray, shape (n_days, n_features)
        Fully normalised feature matrix with no NaNs.
    feature_cols : list[str]
        Column names corresponding to each column of X.
    meta_df : pd.DataFrame
        Metadata rows aligned with X: ``id``, ``date``, and any of
        ``disease_type``, ``sex``, ``age`` present in ``daily_df``.
    """
    df = daily_df.copy()

    # Fill pollen_total NaN (Oct–Dec not measured) with 0 before dropping
    if 'pollen_total' in df.columns:
        df['pollen_total'] = df['pollen_total'].fillna(0)

    # ── Within-patient z-score for HR features → suffix _z ───────────────────
    # NaN can appear here when a patient has only 1 complete day (std = 0),
    # or when the derived feature itself is NaN (e.g. hr_per_100steps on a
    # fully sedentary day). Those rows are dropped below.
    for col in _HR_FEATURES:
        if col not in df.columns:
            continue
        patient_mean = df.groupby('id')[col].transform('mean')
        patient_std  = df.groupby('id')[col].transform('std')
        df[f'{col}_z'] = (df[col] - patient_mean) / patient_std.replace(0, np.nan)

    # ── Collect feature columns (suffixed names) ──────────────────────────────
    step_sleep_cols  = [c for c in _STEP_SLEEP_FEATURES if c in df.columns]
    hr_z_cols        = [f'{c}_z' for c in _HR_FEATURES if f'{c}_z' in df.columns]
    ext_cols         = [c for c in _EXTERNAL_FEATURES if c in df.columns] if include_external else []

    keep_cols = ['id', 'date'] + [c for c in ['disease_type', 'sex', 'age'] if c in df.columns]
    all_needed = keep_cols + step_sleep_cols + hr_z_cols + ext_cols
    subset = df[all_needed].dropna(subset=step_sleep_cols + hr_z_cols + ext_cols)

    # ── Global StandardScaler on step/sleep + external → suffix _sc ──────────
    subset = subset.copy()
    sc_target = step_sleep_cols + ext_cols
    sc_names  = [f'{c}_sc' for c in sc_target]
    if sc_target:
        scaler = StandardScaler()
        subset[sc_names] = scaler.fit_transform(subset[sc_target])

    feature_cols = sc_names + hr_z_cols
    X       = subset[feature_cols].to_numpy(dtype=np.float64)
    meta_df = subset[keep_cols].reset_index(drop=True)

    return X, feature_cols, meta_df


def build_ts_tensor(
    df: pd.DataFrame,
    normalize_hr_within_patient: bool = True,
) -> tuple:
    """Build a multivariate time-series tensor for shape-based clustering (Variation 2).

    Each patient-day becomes a ``(24, 3)`` matrix — 24 hours × 3 signals
    (steps, heartrate, sleep) — normalised as follows:

    * **heartrate** — within-patient z-score across all complete hours, applied
      per-day so the shape of the HR curve is preserved but absolute level is removed.
    * **steps** — global min-max to [0, 1].
    * **sleep** — global min-max to [0, 1] (already 0–60 range but variable).

    Parameters
    ----------
    df : pd.DataFrame
        Hourly sensor data from ``get_complete_patient_days(complete=True)``.
        Required columns: ``id``, ``time``, ``shifted_date``, ``steps``,
        ``heartrate``, ``sleep``.
    normalize_hr_within_patient : bool
        If False, apply global min-max to heartrate as well (for ablation).

    Returns
    -------
    X_ts : np.ndarray, shape (n_days, 24, 3)
        Channel order: [steps, heartrate, sleep].  No NaNs — days with any
        missing value in the 24×3 block are dropped.
    day_index : pd.DataFrame
        ``id`` and ``shifted_date`` for each row in ``X_ts``.
    """
    df = df.copy()

    if 'hour' not in df.columns:
        df['hour'] = df['time'].dt.hour

    # ── Within-patient z-score for heartrate ─────────────────────────────────
    if normalize_hr_within_patient:
        p_mean = df.groupby('id')['heartrate'].transform('mean')
        p_std  = df.groupby('id')['heartrate'].transform('std').replace(0, np.nan)
        df['heartrate_norm'] = (df['heartrate'] - p_mean) / p_std
    else:
        hr_min = df['heartrate'].min()
        hr_rng = df['heartrate'].max() - hr_min
        df['heartrate_norm'] = (df['heartrate'] - hr_min) / hr_rng

    # ── Global min-max for steps and sleep ───────────────────────────────────
    for col in ('steps', 'sleep'):
        col_min = df[col].min()
        col_rng = df[col].max() - col_min
        df[f'{col}_norm'] = (df[col] - col_min) / col_rng if col_rng > 0 else 0.0

    # ── Pivot each signal to (id, shifted_date) × hour ───────────────────────
    signals = {'steps': 'steps_norm', 'heartrate': 'heartrate_norm', 'sleep': 'sleep_norm'}
    pivots  = {}
    for name, col in signals.items():
        piv = (
            df.pivot_table(index=['id', 'shifted_date'], columns='hour', values=col, aggfunc='first')
            .reindex(columns=range(24))
        )
        pivots[name] = piv

    # ── Keep only days with all 24 × 3 values present ────────────────────────
    valid_mask = (
        pivots['steps'].notna().all(axis=1) &
        pivots['heartrate'].notna().all(axis=1) &
        pivots['sleep'].notna().all(axis=1)
    )
    valid_idx = valid_mask[valid_mask].index

    arrays = [pivots[s].loc[valid_idx].to_numpy(dtype=np.float64) for s in signals]
    # Stack to (n_days, 24, 3)
    X_ts = np.stack(arrays, axis=2)

    day_index = (
        pd.DataFrame(index=valid_idx)
        .reset_index()
        .rename(columns={'shifted_date': 'shifted_date'})
    )

    return X_ts, day_index


# ---------------------------------------------------------------------------
# External data loaders
# ---------------------------------------------------------------------------

def load_weather(
    path: str,
    url: str = (
        'https://data.geo.admin.ch/ch.meteoschweiz.ogd-smn/'
        'klo/ogd-smn_klo_h_historical_2020-2029.csv'
    ),
    year: int = 2021,
    daily: bool = False,
) -> pd.DataFrame:
    """Load MeteoSwiss Zürich/Kloten weather data.

    Falls back to ``url`` if ``path`` does not exist, and caches the result
    to ``path`` so subsequent calls are local.

    Parameters
    ----------
    path : str
        Local CSV path (MeteoSwiss format, sep=';', encoding='latin-1').
    url : str
        Remote URL used when ``path`` is missing.
    year : int
        Calendar year to filter to.
    daily : bool
        If False (default), return one row per hour with columns
        ``datetime``, ``date``, ``temp_c``, ``temp_c_min``, ``temp_c_max``,
        ``precip_mm``, ``sunshine_min``, ``humidity_pct``.
        If True, aggregate to one row per calendar day with columns
        ``date``, ``temp_max``, ``temp_mean``, ``temp_min``,
        ``precip_total``, ``sunshine_total``, ``humidity_mean``.

    Returns
    -------
    pd.DataFrame
    """
    try:
        raw = pd.read_csv(path, sep=';', encoding='latin-1')
    except FileNotFoundError:
        raw = pd.read_csv(url, sep=';', encoding='latin-1')
        raw.to_csv(path, sep=';', index=False, encoding='latin-1')

    raw['datetime'] = pd.to_datetime(raw['reference_timestamp'], format='%d.%m.%Y %H:%M')
    raw = raw[raw['datetime'].dt.year == year].copy()

    raw = raw[['datetime', 'tre200h0', 'tre200hn', 'tre200hx',
               'rre150h0', 'sre000h0', 'ure200h0']].copy()
    raw.columns = ['datetime', 'temp_c', 'temp_c_min', 'temp_c_max',
                   'precip_mm', 'sunshine_min', 'humidity_pct']
    raw['date'] = raw['datetime'].dt.floor('D')

    if not daily:
        return raw.reset_index(drop=True)

    return raw.groupby('date').agg(
        temp_max       = ('temp_c_max',   'max'),
        temp_mean      = ('temp_c',       'mean'),
        temp_min       = ('temp_c_min',   'min'),
        precip_total   = ('precip_mm',    'sum'),
        sunshine_total = ('sunshine_min', 'sum'),
        humidity_mean  = ('humidity_pct', 'mean'),
    ).reset_index()


def load_calendar(
    path: str,
    year: int = 2021,
    subdiv: str = 'ZH',
) -> pd.DataFrame:
    """Load (or build) the Zürich public-holiday / school-break calendar.

    If ``path`` does not exist the calendar is built from the ``holidays``
    package and written to ``path`` for future use.

    Parameters
    ----------
    path : str
        Local CSV path.
    year : int
        Calendar year to build if the file is missing.
    subdiv : str
        Swiss canton subdivision code used by the ``holidays`` package.

    Returns
    -------
    pd.DataFrame
        One row per calendar day with columns:

        ``date``, ``is_weekend``, ``is_public_hol``, ``is_school_break``,
        ``official_day_off``, ``is_bridge_day``, ``day_type``,
        ``is_extended_day_off``

        ``day_type`` values: ``'workday'``, ``'weekend'``,
        ``'public_holiday'``, ``'bridge_day'``
    """
    try:
        cal = pd.read_csv(path, parse_dates=['date'])
        return cal
    except FileNotFoundError:
        pass

    import holidays as hol_lib

    public_hols = pd.to_datetime(
        list(hol_lib.country_holidays('CH', subdiv=subdiv, years=year).keys())
    )
    school_breaks = [
        (f'{year}-04-26', f'{year}-05-08'),
        (f'{year}-07-19', f'{year}-08-21'),
        (f'{year}-10-11', f'{year}-10-23'),
        (f'{year}-12-20', f'{year+1}-01-01'),
    ]

    cal = pd.DataFrame({'date': pd.date_range(f'{year}-01-01', f'{year}-12-31', freq='D')})
    cal['is_weekend']     = cal['date'].dt.dayofweek >= 5
    cal['is_public_hol']  = cal['date'].isin(public_hols)
    cal['is_school_break'] = False
    for start, end in school_breaks:
        cal.loc[cal['date'].between(start, end), 'is_school_break'] = True

    cal['official_day_off'] = cal['is_weekend'] | cal['is_public_hol'] | cal['is_school_break']
    prev_off = cal['official_day_off'].shift(1, fill_value=False)
    next_off = cal['official_day_off'].shift(-1, fill_value=False)
    cal['is_bridge_day'] = (
        ~cal['official_day_off'] & ~cal['is_weekend'] & prev_off & next_off
    )

    def _day_type(row):
        if row['is_public_hol']:  return 'public_holiday'
        if row['is_bridge_day']:  return 'bridge_day'
        if row['is_weekend']:     return 'weekend'
        return 'workday'

    cal['day_type']            = cal.apply(_day_type, axis=1)
    cal['is_extended_day_off'] = cal['day_type'] != 'workday'
    cal.to_csv(path, index=False)

    return cal


def load_covid(
    path: str,
    url: str = (
        'https://raw.githubusercontent.com/owid/covid-19-data/'
        'master/public/data/owid-covid-data.csv'
    ),
    year: int = 2021,
    iso_code: str = 'CHE',
) -> pd.DataFrame:
    """Load Switzerland COVID-19 stringency index from OWID.

    Falls back to ``url`` if ``path`` does not exist, filters to ``iso_code``
    and ``year``, and caches the result to ``path``.

    Parameters
    ----------
    path : str
        Local CSV path.
    url : str
        Remote OWID COVID CSV URL.
    year : int
        Calendar year to filter to.
    iso_code : str
        ISO 3166-1 alpha-3 country code.

    Returns
    -------
    pd.DataFrame
        Columns: ``date``, ``stringency_index``,
        ``new_cases_smoothed_per_million``
    """
    try:
        df = pd.read_csv(path, parse_dates=['date'])
        return df
    except FileNotFoundError:
        pass

    raw = pd.read_csv(url)
    df = raw[
        (raw['iso_code'] == iso_code) &
        (raw['date'].between(f'{year}-01-01', f'{year}-12-31'))
    ][['date', 'stringency_index', 'new_cases_smoothed_per_million']].copy()
    df['date'] = pd.to_datetime(df['date'])
    df.to_csv(path, index=False)

    return df


def load_pollen(
    path: str,
    url: str = (
        'https://data.geo.admin.ch/ch.meteoschweiz.ogd-pollen/'
        'pzh/ogd-pollen_pzh_d_historical.csv'
    ),
    year: int = 2021,
) -> pd.DataFrame:
    """Load MeteoSwiss Zürich pollen daily concentrations.

    Falls back to ``url`` if ``path`` does not exist, and caches the result.

    Parameters
    ----------
    path : str
        Local CSV path (MeteoSwiss format, sep=';', encoding='latin-1').
    url : str
        Remote URL used when ``path`` is missing.
    year : int
        Calendar year to filter to.

    Returns
    -------
    pd.DataFrame
        One row per calendar day (Jan–Sep coverage) with columns:

        ``date``, ``alder``, ``birch``, ``hazel``, ``beech``, ``ash``,
        ``oak``, ``grasses``, ``pollen_total``

        All pollen values are in grains/m³.
    """
    try:
        raw = pd.read_csv(path, sep=';', encoding='latin-1')
    except FileNotFoundError:
        raw = pd.read_csv(url, sep=';', encoding='latin-1')
        raw.to_csv(path, sep=';', index=False, encoding='latin-1')

    raw['date'] = pd.to_datetime(raw['reference_timestamp'], format='%d.%m.%Y %H:%M')
    raw = raw[raw['date'].dt.year == year].copy()

    raw = raw.rename(columns={
        'kaalnud0': 'alder',
        'kabetud0': 'birch',
        'kacoryd0': 'hazel',
        'kafagud0': 'beech',
        'kafraxd0': 'ash',
        'kaquerd0': 'oak',
        'khpoacd0': 'grasses',
    })[['date', 'alder', 'birch', 'hazel', 'beech', 'ash', 'oak', 'grasses']].copy()

    raw['date']         = raw['date'].dt.normalize()
    raw['pollen_total'] = raw[['alder', 'birch', 'hazel', 'beech', 'ash', 'oak', 'grasses']].sum(axis=1)

    return raw.reset_index(drop=True)


def load_air_quality(
    path: str,
    year: int = 2021,
    daily: bool = False,
) -> pd.DataFrame:
    """Load BAFU NABEL Zürich-Kaserne air quality data.

    The source file must be downloaded manually from the BAFU NABEL portal
    (https://www.bafu.admin.ch/bafu/en/home/topics/air/state/data/air-pollution--real-time-data/download-data-of-the-national-air-pollution-monitoring-network.html)
    and placed at ``path``.

    Parameters
    ----------
    path : str
        Local CSV path (NABEL format: sep=';', encoding='latin-1',
        5 metadata header rows before the column header).
    year : int
        Calendar year to filter to.
    daily : bool
        If False (default), return one row per hour with columns
        ``datetime``, ``date``, ``o3_ugm3``, ``no2_ugm3``, ``so2_ugm3``,
        ``co_mgm3``, ``pm10_ugm3``, ``pm25_ugm3``, ``ec_ugm3``,
        ``nmvoc_ppm``, ``nox_ugm3``.
        If True, aggregate to one row per calendar day with columns
        ``date``, ``pm25_mean``, ``pm10_mean``, ``o3_mean``, ``no2_mean``,
        ``pm25_max``, ``o3_max``.

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist (no remote fallback available).
    """
    raw = pd.read_csv(path, sep=';', encoding='latin-1', skiprows=5)

    raw['datetime'] = pd.to_datetime(raw['Date/time'], format='%d.%m.%Y %H:%M')
    raw['date']     = raw['datetime'].dt.floor('D')

    raw = raw.rename(columns={
        'O3 [ug/m3]':           'o3_ugm3',
        'NO2 [ug/m3]':          'no2_ugm3',
        'SO2 [ug/m3]':          'so2_ugm3',
        'CO [mg/m3]':           'co_mgm3',
        'PM10 [ug/m3]':         'pm10_ugm3',
        'PM2.5 [ug/m3]':        'pm25_ugm3',
        'EC [ug/m3]':           'ec_ugm3',
        'NMVOC [ppm]':          'nmvoc_ppm',
        'NOX [ug/m3 eq. NO2]':  'nox_ugm3',
    })

    hourly_cols = ['datetime', 'date', 'o3_ugm3', 'no2_ugm3', 'so2_ugm3',
                   'co_mgm3', 'pm10_ugm3', 'pm25_ugm3', 'ec_ugm3',
                   'nmvoc_ppm', 'nox_ugm3']
    raw = raw[raw['datetime'].dt.year == year][hourly_cols].reset_index(drop=True)

    if not daily:
        return raw

    return (
        raw.groupby('date')
        .agg(
            pm25_mean = ('pm25_ugm3', 'mean'),
            pm10_mean = ('pm10_ugm3', 'mean'),
            o3_mean   = ('o3_ugm3',   'mean'),
            no2_mean  = ('no2_ugm3',  'mean'),
            pm25_max  = ('pm25_ugm3', 'max'),
            o3_max    = ('o3_ugm3',   'max'),
        )
        .reset_index()
    )
