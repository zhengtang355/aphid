# Dataset Description

The dataset is designed for bean aphid (*Aphis craccivora* Koch) parturition behavior recognition from macro-monitoring videos.

## Behavior Classes

| Code label | Paper label | Description |
| --- | --- | --- |
| `nobirth` | `non-parturition` | No nymph emergence is observed. |
| `start` | `parturition onset` | The early stage after nymph emergence begins. |
| `birth` | `ongoing parturition` | The nymph has emerged and remains visible during parturition. |

## Suggested Directory Structure

```text
dataset/
  train/
    nobirth/
    start/
    birth/
  val/
    nobirth/
    start/
    birth/
```

Each sample folder or video clip should contain an individual aphid behavior segment. The original experiments used 32 uniformly sampled frames resized to 224 x 224 pixels.

## Detection Annotations

For video-level visualization, CVAT XML annotations can be used to provide aphid bounding boxes and track IDs. 

