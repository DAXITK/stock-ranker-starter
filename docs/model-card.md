# Model card

## Intended use
Educational ML deployment demonstration, not an investment recommendation.

## Data
Public demonstration uses invented synthetic price/volume data. See metadata.json
for provenance, snapshot date, model version and actual training timestamp.

## Model and evaluation
Regression scores rank tickers by predicted five-observation forward return.
Chronological holdout with purged training labels. Report MAE, a zero-return
baseline and daily rank correlation. Synthetic results do not show market skill.

## Limitations
Fixed ticker universe, overlapping targets and dated snapshot. No live refresh,
point-in-time fundamentals, drift detection or delayed accuracy monitoring.

## Operations
Deploy trusted bundles. Checksums detect accidental corruption, not adversarial
replacement of artifacts and metadata. HTTP counters reset on process restart.

## History
See HISTORY.md for retrospective reconstruction and timestamp disclosure.
