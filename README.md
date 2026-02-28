# PulmoSight-CNN

A deep-learning histopathology classifier that distinguishes normal lung tissue from adenocarcinoma and squamous cell carcinoma.

## Project Structure

- `src/data/`: dataset loading/splitting, transforms, and visualization helpers.
- `src/models/`: CNN architecture definition.
- `src/train.py`: end-to-end training loop with artifact generation.
- `src/evaluate.py`: validation metrics, confusion matrix, classification report.
- `configs/`: YAML-based training and path configuration.

## Expected Dataset Layout

```text
data/lung_histopath/
  normal/
  adenocarcinoma/
  squamous_cell_carcinoma/
```

## Run Training

```bash
python src/train.py --config configs/config.yaml
```

Artifacts are written to `outputs/` by default:
- best model weights
- class distribution plot
- sample batch preview
- training curves
- confusion matrix
- classification report
- metrics yaml
