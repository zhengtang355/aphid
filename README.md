# PC-SGMSlowFast

This repository provides code for bean aphid (*Aphis craccivora* Koch) parturition behavior recognition in macro-monitoring videos. The method uses a two-stage framework: YOLOv8 first localizes individual aphids, and the improved SlowFast model PC-SGMSlowFast then classifies individual behavior clips into three classes.

## Behavior Classes

- `non-parturition`
- `parturition onset`
- `ongoing parturition`

The internal training labels in the code may use `nobirth`, `start`, and `birth`, corresponding to the three classes above.

## Repository Structure

```text
configs/        Training configuration files
data_tools/     Data conversion, frame extraction, and crop generation scripts
train/          Main training script for DESASlowFast-TDEM and baseline models
eval/           Evaluation and model-complexity scripts
inference/      Video inference and result visualization scripts
visualization/  Attention and dual-end mask visualization scripts
baselines/      Baseline video recognition model scripts
figures/        Model and workflow diagrams
```

## Requirements

The experiments were conducted with Python 3.9.23, PyTorch 2.8.0, CUDA 12.6, and an NVIDIA GeForce RTX 2080 GPU. Main dependencies are listed in `requirements.txt`.

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Training

Example command for training PC-SGMSlowFast:

```bash
python train/train_all_baselines_config.py --config configs/baseline_config_3class_slowfast_uniform32_desa_tda.json
```

Please update dataset paths in the configuration file before training.

## Evaluation

Example command for evaluating a trained checkpoint:

```bash
python eval/eval_checkpoint_confusion_matrix.py --checkpoint path/to/best_model_by_mAP.pth --config configs/baseline_config_3class_slowfast_uniform32_desa_tda.json
```

Model parameters and FLOPs can be computed with scripts in `eval/`.

## Inference and Visualization

For XML-annotation-based multi-target visualization:

```bash
python inference/infer_slowfast_xml_after_default.py ^
  --video path/to/video.mp4 ^
  --xml path/to/annotations.xml ^
  --checkpoint path/to/best_model_by_mAP.pth ^
  --output_dir outputs ^
  --default_seconds 15 ^
  --window_seconds 15
```

For direct label visualization without model inference:

```bash
python inference/visualize_labeled_segments.py ^
  --video path/to/video.mp4 ^
  --xml path/to/annotations.xml ^
  --output outputs/annotated.mp4 ^
  --durations 15,20,15 ^
  --labels nobirth,start,birth
```

## Data

The bean aphid parturition behavior dataset is released separately. See `DATASET.md` for the expected data organization and label definitions.

## Citation

If you use this code or dataset, please cite the related paper.
