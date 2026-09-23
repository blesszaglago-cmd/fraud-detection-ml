# Rules

## What to Do
- Use scikit-learn for modeling
- Use imbalanced-learn for SMOTE
- Always scale Amount and Time
- Use stratified split to preserve class ratio
- Evaluate with precision, recall, F1, ROC-AUC
- Document every function with a docstring

## What to Avoid
- Do not train on imbalanced data without handling it
- Do not use accuracy as the main metric (misleading with imbalance)
- Do not leak test data into training
- Do not hardcode file paths — use pathlib
- Do not skip random_state

## Error Handling
- If CSV is missing, print clear error
- If columns are missing, list which ones
- If model file missing, train a new one

## Constraints
- Keep code readable
- Every script runnable on its own
- No external APIs
- Random seed = 42 everywhere