---
name: datarobot-model-explainability
description: >
  Tools and guidance for model explainability, prediction explanations, feature impact analysis,
  SHAP values, SHAP distributions, anomaly assessment, and model diagnostics. Use when analyzing
  model explanations, feature impact, SHAP values, SHAP distributions, anomaly assessment, or
  diagnosing model behavior.
---

# DataRobot Model Explainability Skill

This skill provides comprehensive guidance for understanding model decisions, analyzing prediction
explanations, and interpreting model behavior using DataRobot's explainability APIs.

> **SDK version**: Targets `datarobot>=3.4.0`. Use `from datarobot.insights import ShapMatrix, ...`
> with `entity_id=model_id` — not legacy `datarobot.models.ShapMatrix` (`project_id` / `dataset_id`).
> `ShapMatrix`, `ShapImpact`, `ShapPreview`, and `ShapDistributions` are the canonical SHAP API.
> The older `dr.PredictionExplanations` (XEMP-based) remains available but is the secondary path.

---

## Quick Start

| Goal | API to use | Prerequisites |
|------|-----------|---------------|
| SHAP values for all features, all rows | `ShapMatrix.create(entity_id=model_id)` | None - universal SHAP |
| Per-row top-feature explanations | `ShapPreview.create(entity_id=model_id)` | None |
| Aggregated feature importance via SHAP | `ShapImpact.create(entity_id=model_id)` | None |
| SHAP value distributions across features | `ShapDistributions.create(entity_id=model_id)` | None |
| SHAP for a filtered segment | `dr.DataSlice.create(...)` + `ShapMatrix.create(..., data_slice_id=...)` | Data slice definition |
| XEMP-based prediction explanations | `dr.PredictionExplanations.create(...)` | Feature Impact; PE initialization; dataset uploaded |
| Anomaly explanations (time series) | `AnomalyAssessmentRecord.compute(project_id, model_id, ...)` | Anomaly model |
| ROC / lift / confusion (insights) | `RocCurve` / `LiftChart` / `ConfusionMatrix`.create(entity_id=model_id) | Validation data |
| ROC / lift / confusion (Model helpers) | `model.get_roc_curve()` / `model.get_lift_chart()` / `model.get_confusion_chart()` | Validation data |

**Universal SHAP is the preferred path** - no dataset pre-upload or Feature Impact step required.

## When to use this skill

Use this skill when you need to:
- Explain DataRobot leaderboard model predictions with SHAP or XEMP
- Compute full SHAP matrices, compact SHAP previews, SHAP impact, or SHAP distributions
- Compare global and local feature contributions
- Filter explainability results to a data slice or cohort
- Investigate time series anomaly explanations
- Retrieve model diagnostics such as ROC curves, lift charts, confusion charts, and feature effects

## Key capabilities

### 1. Modern SHAP insights

- Use `datarobot.insights` for `ShapMatrix`, `ShapPreview`, `ShapImpact`, and `ShapDistributions`
- Compute universal SHAP without Feature Impact or prediction-explanation initialization
- Export SHAP matrices to pandas DataFrames or CSV

### 2. Legacy and model-specific explanation paths

- Use XEMP `dr.PredictionExplanations` when required by model type or stakeholder expectations
- Handle multiclass and clustering explanation modes
- Keep deployment-time prediction explanations in `datarobot-predictions`

### 3. Diagnostics and specialized insights

- Apply `dr.DataSlice` filters to SHAP insights
- Use anomaly assessment records for time series anomaly explanations
- Retrieve diagnostics via `datarobot.insights` (`RocCurve`, `LiftChart`, `ConfusionMatrix`) or Model helpers
- Retrieve partial dependence via `model.get_feature_effect()` (`FeatureEffects` objects)

---

## Setup

```python
import os
import datarobot as dr
from datarobot.insights import ShapMatrix, ShapImpact, ShapPreview, ShapDistributions

dr.Client(
    token=os.environ["DATAROBOT_API_TOKEN"],
    endpoint=os.environ.get("DATAROBOT_ENDPOINT", "https://app.datarobot.com/api/v2"),
)
```

---

## Core API: `datarobot.insights`

### ShapMatrix - raw SHAP values per row

`ShapMatrix` gives you the full SHAP value matrix: one row per prediction, one column per feature.

```python
import pandas as pd
from datarobot.insights import ShapMatrix

model_id = "YOUR_MODEL_ID"

# Compute on validation partition (default), wait for result.
result = ShapMatrix.create(entity_id=model_id)

# Export: matrix/columns are already on the result object
df = pd.DataFrame(result.matrix, columns=result.columns)

# Or re-fetch from the API (classmethods — not instance methods):
df = ShapMatrix.get_as_dataframe(entity_id=model_id, source="validation")
csv = ShapMatrix.get_as_csv(entity_id=model_id, source="validation")

# Compute on a different partition, non-blocking
job = ShapMatrix.compute(entity_id=model_id, source="holdout")
result = job.get_result_when_complete()

# Compute on an external dataset.
dataset = dr.Dataset.upload("./data/scoring_data.csv")
job = ShapMatrix.compute(
    entity_id=model_id,
    source="externalTestSet",
    external_dataset_id=dataset.id,
)
result = job.get_result_when_complete()

all_matrices = ShapMatrix.list(entity_id=model_id)
```

**When to use**: You need per-row, per-feature SHAP values for downstream analysis, export, or
custom visualization. See `references/shap_api_reference.md` for attributes and limitations.

---

### ShapImpact - aggregated feature importance

`ShapImpact` aggregates the SHAP matrix into overall feature importance scores.

```python
from datarobot.insights import ShapImpact

# Compute (training partition only for ShapImpact)
job = ShapImpact.compute(entity_id=model_id, source="training")
result = job.get_result_when_complete()

for item in result.shap_impacts:
    if isinstance(item, dict):
        print(
            f"{item['feature_name']}: {item['impact_normalized']:.4f} (normalized), "
            f"{item['impact_unnormalized']:.4f} (raw)"
        )
    else:
        name, norm, unnorm = item
        print(f"{name}: {norm:.4f} (normalized), {unnorm:.4f} (raw)")
```

**When to use**: Global feature importance. Prefer over `model.get_feature_impact()` when you
want SHAP-consistent importance (same methodology as local SHAP explanations).

---

### ShapPreview - per-row top-feature explanations

`ShapPreview` returns top-N features and their SHAP values per prediction row - compact format
suited for display or alerting.

```python
from datarobot.insights import ShapPreview

# Compute on validation
result = ShapPreview.create(entity_id=model_id)

for row in result.previews:
    print(row["row_index"], row["prediction_value"], row["preview_values"])

# Compute on external dataset
dataset = dr.Dataset.upload("./data/scoring_data.csv")
job = ShapPreview.compute(
    entity_id=model_id,
    source="externalTestSet",
    external_dataset_id=dataset.id,
)
result = job.get_result_when_complete()
```

**When to use**: Human-readable "top drivers" view per prediction. Lighter than full ShapMatrix.

---

### ShapDistributions - SHAP value distributions per feature

`ShapDistributions` shows how SHAP values are distributed across rows for each feature.

```python
from datarobot.insights import ShapDistributions

result = ShapDistributions.create(entity_id=model_id)

print(result.total_features_count)
for feature in result.features:
    print(feature["feature_name"])
```

**When to use**: Understanding which features have high variance in their SHAP contributions
(i.e., features with inconsistent influence across the dataset).

---

## Secondary path: XEMP Prediction Explanations

Use `dr.PredictionExplanations` when XEMP explanations are specifically required (e.g., certain
regulatory contexts, or when SHAP is unavailable for the model type).

**Prerequisites** (all required before calling `.create()`):
1. Feature Impact must be computed: `model.request_feature_impact()` and wait
2. Prediction explanations initialized: `dr.PredictionExplanationsInitialization.create(...)`
3. Scoring dataset uploaded to the AI Catalog

```python
import datarobot as dr

project_id = "YOUR_PROJECT_ID"
model_id = "YOUR_MODEL_ID"
model = dr.Model.get(project=project_id, model_id=model_id)

model.request_feature_impact().wait_for_completion()
dr.PredictionExplanationsInitialization.create(project_id=project_id, model_id=model_id)

dataset = dr.Dataset.upload("./data/scoring_data.csv")
pe_job = dr.PredictionExplanations.create(
    project_id=project_id,
    model_id=model_id,
    dataset_id=dataset.id,
    max_explanations=5,      # top N features per row
    threshold_high=0.5,      # only explain rows with prediction >= threshold
    threshold_low=0.1,       # only explain rows with prediction <= threshold
)

pe_obj = pe_job.get_result_when_complete()
for row in pe_obj.get_rows():
    print(row.prediction)
    for explanation in row.prediction_explanations:
        print(f"  {explanation['feature']}: strength={explanation['strength']:.4f}")

# Export
pe_obj.download_to_csv("explanations.csv")
df = pe_obj.get_all_as_dataframe()
```

For parameters, multiclass modes, and exposure-adjusted predictions, see
`references/xemp_pe_reference.md`.

### Multiclass / clustering - specify classes to explain

```python
pe_job = dr.PredictionExplanations.create(
    project_id=project_id,
    model_id=model_id,
    dataset_id=dataset.id,
    mode=dr.models.ClassListMode(["class_a", "class_b"]),  # which classes to explain
)
```

---

## Data slices for filtered insights

Use `dr.DataSlice` when the user asks to explain model behavior for a segment, such as a
region, product line, target class, or high-risk cohort. Pass the resulting `data_slice_id` into
the `datarobot.insights` SHAP APIs.

```python
import datarobot as dr
from datarobot.insights import ShapMatrix

project_id = "YOUR_PROJECT_ID"
model_id = "YOUR_MODEL_ID"

data_slice = dr.DataSlice.create(
    name="high_income_customers",
    filters=[{"operand": "income", "operator": ">", "values": 100000}],
    project=project_id,
)

shap_matrix = ShapMatrix.create(
    entity_id=model_id,
    source="validation",
    data_slice_id=data_slice.id,
)
```

---

## Anomaly assessment (time series models)

For time series anomaly detection models, use `AnomalyAssessmentRecord`.

```python
from datarobot.models.anomaly_assessment import AnomalyAssessmentRecord

project_id = "YOUR_PROJECT_ID"
model_id = "YOUR_MODEL_ID"

record = AnomalyAssessmentRecord.compute(
    project_id=project_id,
    model_id=model_id,
    backtest=0,           # backtest index (int) or "holdout"
    source="validation",  # "training" or "validation" only
    series_id=None,       # required for multiseries projects
)

preview = record.get_predictions_preview()
anomalous_regions = preview.find_anomalous_regions()

explanations = record.get_explanations_data_in_regions(regions=anomalous_regions)

# Reuse an existing record when possible
records = AnomalyAssessmentRecord.list(project_id=project_id, model_id=model_id)

# Latest explanations for the most anomalous records (no date filters)
latest = record.get_latest_explanations()

# Date-range explanations (two of start_date, end_date, or points_count required)
ranged = record.get_explanations(
    start_date="2024-01-01T00:00:00.000000Z",
    end_date="2024-06-01T00:00:00.000000Z",
)
```

---

## Model diagnostics

Use the same `entity_id=model_id` pattern as SHAP insights. `FeatureEffects` / partial dependence
is still retrieved through Model helpers (not in `datarobot.insights`).

### Insights diagnostics (preferred — matches SHAP API)

```python
from datarobot.insights import RocCurve, LiftChart, ConfusionMatrix

model_id = "YOUR_MODEL_ID"

# Blocking compute + retrieve (default source='validation')
roc = RocCurve.create(entity_id=model_id)
print(roc.auc, len(roc.roc_points))

lift = LiftChart.create(entity_id=model_id)
print(lift.bins)

confusion = ConfusionMatrix.create(entity_id=model_id)
print(confusion.global_metrics, confusion.confusion_matrix_data)

# Optional: non-blocking compute, or pass data_slice_id=... from dr.DataSlice
job = RocCurve.compute(entity_id=model_id)
roc = job.get_result_when_complete()
```

### Model helpers (alternative)

```python
model = dr.Model.get(project=project_id, model_id=model_id)

roc = model.get_roc_curve(source="validation")
lift = model.get_lift_chart(source="validation")
confusion = model.get_confusion_chart(source="validation")

# Feature Impact (non-SHAP) and Feature Effects (partial dependence for top features)
fi = model.get_feature_impact()
feature_effects = model.get_feature_effect(source="validation")
```

---

## Interpreting SHAP values

- **Positive value**: feature pushes prediction higher than baseline
- **Negative value**: feature pushes prediction lower than baseline
- **Magnitude**: size of influence; larger absolute value = stronger effect
- **Sum**: all SHAP values for a row sum to `prediction - base_value`
- **`base_value`**: the model's mean prediction (the "no information" baseline)

Example: if `base_value = 0.35` and a row's prediction is `0.72`, the row's SHAP values sum to
`0.37`. A feature with SHAP `+0.20` contributed 20 percentage points above baseline.

When `link_function = "logit"`, SHAP values are in log-odds space. Use `exp(shap)` for
probability-space interpretation.

---

## Decision guide

```
Task: explain predictions
    |
    - Need all features + all rows?     -> ShapMatrix.create(entity_id=model_id)
    - Need top-N features per row?      -> ShapPreview.create(entity_id=model_id)
    - Need aggregated importance?       -> ShapImpact.compute(entity_id=model_id, source='training')
    - Need feature SHAP distributions?  -> ShapDistributions.create(entity_id=model_id)
    - Need a segment/cohort only?       -> dr.DataSlice + data_slice_id
    - XEMP required (regulatory/type)?  -> dr.PredictionExplanations.create(...)
    - Time series / anomaly model?      -> AnomalyAssessmentRecord.compute(project_id, model_id, ...)
```

---

## Common errors

| Error | Cause | Fix |
|-------|-------|-----|
| `SHAP not available for this model` | Blender or >1000-feature model | Use XEMP PE instead |
| `Feature Impact not computed` | PredictionExplanations prerequisite missing | Run `model.request_feature_impact()` and wait |
| Missing `PredictionExplanationsInitialization` | PE not initialized | Call `PredictionExplanationsInitialization.create()` |
| `source='holdout'` fails | Holdout not unlocked | Unlock holdout in project settings first |
| Empty `previews` | No rows in partition | Check partition contains data |

---

## Reference files

- `references/shap_api_reference.md` - full parameter signatures for ShapMatrix, ShapImpact, ShapPreview, ShapDistributions
- `references/xemp_pe_reference.md` - PredictionExplanations and PredictionExplanationsInitialization parameter reference
- `scripts/compute_shap_matrix.py` - compute and export ShapMatrix to CSV or DataFrame

## Resources

- [datarobot.insights API reference](https://datarobot-public-api-client.readthedocs-hosted.com/en/latest-release/insights.html)
- [SHAP insights user guide](https://datarobot-public-api-client.readthedocs-hosted.com/en/latest-release/reference/modeling/insights/shap_insights.html)
- [Prediction Explanations user guide](https://datarobot-public-api-client.readthedocs-hosted.com/en/latest-release/reference/modeling/insights/prediction_explanations.html)
- [DataRobot Prediction Explanations product docs](https://docs.datarobot.com/en/docs/modeling/analyze-models/understand/pred-explain/predex-overview.html)
