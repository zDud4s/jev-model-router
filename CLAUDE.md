# jev-model-router

## Model-agnostic by design

This project must keep working, unedited, however many models are released. A new
model is data, never a code change:

- No model, vendor or version is named in `jev_model_router/` except in comments that
  record a measurement. Everything model-specific lives in config and data files
  (profiles, cards, anchors, calibration output).
- A newly released model must have a clear, mechanical path to a card and into
  Jev routing: discovered by the catalog, given a prior from evidence, refined by
  anchors and logged outcomes. "Hand-edit the source when X ships" is a design bug.
- Mechanisms are judged by whether they generalise to model N+1: prefer rules
  over the requirements and the evidence (dominance, caps, fitted scales) to rules
  about particular models.
