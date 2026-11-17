# CF-Twins

Core-code supplement for Counterfactual Twins.

## Layout

- `framework/`: representation preprocessing, atomic functions and their calling interface.
- `output/`: the LLM-generated Q1/Q2/Q3 feature programs and joint classifier.

## Calling boundary

The LLM generates Python programs. Each generated `extract(inputs, atoms)` function
calls the supplied atomic-function dictionary and assembles its own features.
The framework supplies geometric evidence; feature composition and classification
remain in the generated code under `output/`.

```python
from framework.api import atoms_for
from output import q1

features = q1.extract(inputs, atoms_for('q1', trace_callback))
```

Use `q2` or `q3` for the other extractors. The host supplies prepared input views
and a zero-argument callback that records the atomic trace; its return value is ignored.

The generated classifier exposes `fit_multiclass(features, labels)` and
`predict_multiclass(features, state)`. Its computation is preserved from the frozen
main-experiment program; descriptive fit metadata is omitted.
