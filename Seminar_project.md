# Interactive DS in Digital Health — Seminar Homework Project

> This file is the single source of truth for the homework project. Keep it updated as analysis decisions are made.

---

## Assignment Overview

**Course:** Interactive Data Science in Digital Health (3 ECTS block seminar, UZH IfI MSc)
**Instructors:** Prof. Jürgen Bernard (IVDA Group), Dr. Michaela Benk, Prof. Viktor von Wyl (healthcare)
**Student:** Sarunchana Yongchaiwathana (Ping)
**Deadline:** 31 May 2026, 23:59
**Submission:** OLAT → Deliverables → "Homework" (PDF or Word report)

**Goal:** Plan, conduct, and document an analysis of wearable sensor data following the IVDA methodology: Data → Analysis Tool → Pattern → Context → Knowledge.

---

## Dataset

### Background
Study on persons with an unspecified chronic disease (symptoms: fatigue + deteriorating physical fitness; disease progresses from early to late stage, has a "fast progressing" form — likely Multiple Sclerosis or similar). Data collected with commercial wearable sensors during:
- **Intervention phase:** 1–3 weeks (varies per participant), emphasis on improving physical fitness
- **Follow-up phase:** 4 weeks (fixed for all participants)
- Some participants dropped out early.
- Data assumed collected in **Zurich, Switzerland**.
- Approximate date range: **April–September 2021**.

### File 1: Hourly Sensor Data
**Location:** `Analysis TAsk/Hourly Sensor Data/RHourly_{id}.csv` (44 files, one per participant)
**Total rows across all files:** ~59,296 (hourly granularity)

| Column | Type | Description | Notes |
|--------|------|-------------|-------|
| `time` | string | Datetime of measurement | Format: `YYYY-MM-DD HH:MM:SS`. Must parse to datetime. |
| `steps` | int | Steps in that hour | Range [0, 6123]. Mean 296. Heavily right-skewed (median 75). Many zero-hours. |
| `sleep` | int (byte) | Minutes of sleep in that hour | Range [0, 60]. Bimodal: mostly 0 or 60. |
| `heartrate` | string | Hourly average BPM | **Stored as string — must cast to float.** Some rows may be empty string (missing). |

The participant `id` is encoded in the filename (e.g., `RHourly_1120.csv` → id = 1120).

### File 2: Clinical Markers
**Location:** `Analysis TAsk/ClinicalMarkers_final.csv`
**Rows:** 44 (one per participant)

| Column | Type | Description | Notes |
|--------|------|-------------|-------|
| `Record` | int | Record number | Not relevant for analysis. |
| `Age` | int | **Birth year**, not age | Range [1958, 2005]. Derive age as `2021 - Age`. |
| `sex` | string | Sex | "Female" (29), "Male" (15) |
| `disease.type` | string | Disease stage/type | "Early Disease Stage" (18), "Fast Disease Progression" (7), "Late Disease Stage" (19) |
| `Id` | int | Participant ID | **Links to sensor filename.** e.g., Id=3389 → `RHourly_3389.csv` |

### Data Quality Issues to Address at Load Time
1. `heartrate` is a string column — cast to float; empty strings become NaN.
2. `Age` is birth year — compute `age = 2021 - Age`.
3. Intervention phase boundaries are **not labeled** in the data — must be inferred (e.g., by participant's earliest date + known duration range) or treated as unknown.
4. Some participants have fewer rows due to early dropout.
5. Column names in ClinicalMarkers use mixed conventions (`disease.type` with dot, leading space on `sex`) — clean on load.

---

## Tech Stack

- **Analysis:** Python, Jupyter Notebook
- **Core libraries:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`, `sklearn` , `plotly`
- **Interactive visualizations:** TBD — candidates: Plotly, Altair, or exported to a separate tool. Analysis and aggregation happen in the notebook; interactive plots may live elsewhere.
- **Report writing:** TBD (Overleaf + bibtex preferred for clean references)

---

## External Datasets (all 3 are mandatory/planned)

Merging at least one external dataset is a **mandatory assignment requirement**. We are using all three.

| Dataset | What it adds | Merge key | Source |
|---------|-------------|-----------|--------|
| **Zurich weather** | Temperature, sunshine hours, precipitation — directly relevant to outdoor activity (steps) | `date` or `datetime` | MeteoSwiss open data |
| **Swiss public holidays + school holidays 2021** | Binary flag: is this a holiday/weekend? Affects step count patterns. | `date` | Swiss federal calendar |
| **COVID-19 stringency index (Switzerland, 2021)** | Apr–May 2021 had restrictions that lifted through summer — directly affects mobility and step counts | `date` | Our World in Data / Oxford COVID-19 Government Response Tracker |

Merge strategy: all external data joins to the sensor data on the `date` component of the `time` column.

---

## Task Plan: Doing BOTH A and B

The assignment asks for either A or B. If both are submitted, the stronger is graded. We are doing both and starting with A.

---

## Choice A: Data Exploration (starting here)

**Requirement:** At minimum 3 entirely different findings. Each finding must have its own visualization that visually prioritizes that finding. Follow "overview first, zoom and filter, details on demand."

### Planned Findings (to be confirmed during EDA)

**Finding 1 — Activity patterns by time of day and disease stage**
Do patients with different disease stages show different daily activity rhythms? Expect early-stage patients to have more distinct activity peaks; late-stage may be flatter. Visualization: heatmap of average hourly steps by hour-of-day × disease stage.

**Finding 2 — Effect of weather/season on physical activity**
Do warmer, sunnier days correlate with higher step counts? Zurich weather data provides the context. Visualization: scatter or binned line plot of daily steps vs. temperature, colored by disease stage.

**Finding 3 — Sleep and heart rate patterns across the study period**
How do sleep minutes and resting heart rate evolve over the study timeline (intervention vs. follow-up)? Look for improvement signal post-intervention. Visualization: time series (smoothed) of group-average sleep and heart rate by week.

**Optional Finding 4 — COVID stringency + step counts**
Did the May 2021 relaxation of Swiss COVID restrictions produce a visible uplift in steps? Visualization: step count trend overlaid with stringency index changes.

**Bonus — Clustering participants by activity profile**
Apply clustering (e.g., k-means on aggregated daily step/sleep/HR features) to identify activity phenotypes. Do clusters align with disease stage? Visualization: 2D PCA scatter plot of participant clusters, colored by disease stage. This earns the **+0.25 grade bonus**.

---

## Choice B: Confirmatory Analysis (after A)

**Requirement:** At minimum 3 entirely different hypotheses, each with PICO framework, statistical method, and a visualization.

### Planned Hypotheses (to be defined in detail when we get here)

**Hypothesis 1 — Disease stage and daily step count**
H: Participants with Late Disease Stage have significantly lower daily step counts than Early Stage participants.
PICO sketch: P=all participants, I=ignore intervention phases, C=Early vs. Late disease stage, O=mean daily steps.

**Hypothesis 2 — Intervention effect on heart rate**
H: Resting heart rate decreases during the follow-up phase compared to the intervention phase.
PICO sketch: P=all completers, I=intervention phase, C=follow-up phase, O=mean daily resting HR.

**Hypothesis 3 — Weather and activity**
H: Daily step count is positively correlated with daily max temperature in Zurich.
PICO sketch: P=all participants, I=none, C=temperature quartiles, O=daily steps (aggregated).

---

## Report Structure

Max **6 pages of written text** (figures, tables, references excluded from page count). Include as many figures as possible — heavily weighted in grading.

| # | Section | Notes |
|---|---------|-------|
| 0 | Abstract | Background / objective / methods / results / conclusions |
| 1 | Introduction | Context, motivation, methods used, results summary, validation approach |
| 2 | Background and Related Work | Domain background + related studies. **≥5 references required.** |
| 3 | Methods | How results were reached; methods/models/experiment designs |
| 4 | Data Characterization | Dataset size, attributes, quality, relation to disease background |
| 5 | Implementation and Software Use | Tech stack description (a few sentences) |
| 6 | Results | Tables, numbers, visualizations — separate from methods |
| 7 | Discussion | Align/disagree with other studies; critical reflection |
| 8 | Limitations | Conditions where results may not hold; data limitations |
| 9 | Reflections | Personal experience during elaboration; process emphasis |
| 10 | Conclusions | Advice to domain experts; future work |
| 11 | References | Use Overleaf + bibtex or Word — no hand-crafted list |
| 12 | Declaration of AI Use | Goals, prompts, outputs, impact on work |

---

## Claude's Role

Support and planning only. Claude helps with:
- Analysis planning and hypothesis formulation
- EDA strategy and visualization design
- Suggesting and sourcing external datasets
- Reviewing code and draft sections
- Structuring the report

Claude does **not** write the report or run the analysis autonomously unless explicitly asked.

---

## Status

- [x] Assignment requirements extracted
- [x] Dataset structure documented
- [x] Tech stack decided
- [x] External datasets identified
- [x] Choice A findings planned
- [ ] Choice A: EDA notebook created and run
- [ ] Choice A: External datasets fetched and merged
- [ ] Choice A: Findings confirmed and visualizations finalized
- [ ] Choice B: Hypotheses defined in detail (PICO + stats)
- [ ] Choice B: Analysis run and results visualized
- [ ] Report drafted
- [ ] Report reviewed and submitted to OLAT
