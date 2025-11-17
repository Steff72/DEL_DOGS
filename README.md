# Dog Breed Classification (Stanford Dogs)

This repository contains the experiments for the DEL mini‑challenge: building a robust classifier that distinguishes the 120 breeds that appear in the [Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs). Almost the entire workflow lives in `dogs_classifier.ipynb`, which documents every step from exploratory analysis to the final transfer‑learning run.

## Repository map
- `dogs_classifier.ipynb` – main research notebook with EDA, training, and evaluation.
- `helper_utils.py` – small helpers to inspect the dataset (directory tree, sample grids, distributions) and to plot TensorBoard logs.
- `data/` – local copy of the dataset (see instructions below).
- `models/` – exported checkpoints, including the best `final_coatnet0.pt` and the associated `class_to_idx.json`.
- `runs/` – TensorBoard logs for most experiments.
- `requirements.txt` – frozen dependency list used during development.

## Setup
1. **Python environment** – the project was built with Python 3.11 + PyTorch 2.8 (Apple Silicon friendly). Create and activate a virtual environment of your choice.
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Dataset placement** – download and extract the Stanford Dogs dataset so that each breed folder lives under `data/raw/Images/<class-name>` (WordNet IDs prefixed in the folder names). The notebook also stores stratified splits under `data/splits` as convenience copies.

## Notebook workflow
1. **EDA** – inspect the directory structure, label distribution, and representative samples. `helper_utils.print_data_folder_structure` and `plot_group_overview_grid` help verify that all 120 classes are present.
2. **Preprocessing** – compute dataset statistics and build transformation pipelines. Earlier baselines resized to 128×128 with dataset normalization, whereas later transfer‑learning runs switch to 224×224 with ImageNet normalization to match pretrained weights.
3. **Data splitting** – the raw dataset is split into 70 % train, 15 % validation, and 15 % test via `torch.utils.data.random_split`. Custom `SubsetWithTransform` wrappers keep augmentation exclusive to the training subset.
4. **Baselines & classic CNNs** – start with small CNNs and progressively add improvements (dropout, batch norm, He initialization). Hyperparameters are validated with manual grid searches and 5‑fold cross‑validation where feasible.
5. **Transfer learning** – fine‑tune ResNet18 variants and finally CoAtNet‑0 with discriminative learning rates, AdamW, label smoothing, heavy augmentation (RandomResizedCrop/ColorJitter/Rotation + mixup & cutmix), and cosine LR decay.
6. **Final model** – the two‑phase fine‑tuning of `coatnet_0_rw_224.sw_in1k` reaches **92.3 % test accuracy (loss 1.04)** after training on the combined train+val split and evaluating once on the untouched 15 % test split. The checkpoint is saved as `models/final_coatnet0.pt`.

You can reproduce every step by running each section of the notebook sequentially. TensorBoard logs are written to `runs/<experiment-name>`; view them via `tensorboard --logdir runs`.

## Using the best model without the notebook
All artefacts needed for inference are in `models/`:
- `final_coatnet0.pt` – contains the weights plus a `config` dict (`model_name`, `img_size`, `num_classes`).
- `class_to_idx.json` – mapping used during training; invert it to display readable class names.

The snippet below shows how to load the checkpoint, prepare an image, and obtain predictions entirely in a Python script or REPL.

```python
import json
from pathlib import Path

import torch
import timm
from PIL import Image
from torchvision import transforms

DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available() else "cpu"
)

# Load checkpoint and rebuild the model
ckpt = torch.load("models/final_coatnet0.pt", map_location=DEVICE)
cfg = ckpt["config"]
model = timm.create_model(
    cfg["model_name"],          # coatnet_0_rw_224.sw_in1k
    pretrained=False,
    num_classes=cfg["num_classes"],
).to(DEVICE)
model.load_state_dict(ckpt["state_dict"])
model.eval()

# Index ↔︎ label lookup
with open("models/class_to_idx.json") as fp:
    idx_to_class = {idx: cls for cls, idx in json.load(fp).items()}

# Same normalization that was used during fine-tuning
inference_tfms = transforms.Compose([
    transforms.Resize(cfg["img_size"]),
    transforms.CenterCrop(cfg["img_size"]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def predict(image_path: str, topk: int = 5):
    img = Image.open(image_path).convert("RGB")
    tensor = inference_tfms(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)
    conf, idx = probs.topk(topk)
    results = []
    for score, class_idx in zip(conf[0], idx[0]):
        raw_name = idx_to_class[class_idx.item()]
        pretty = raw_name.split("-", 1)[-1].replace("_", " ")
        results.append((pretty.title(), float(score)))
    return results

# Example usage
for label, score in predict("pics/happy_dogs.png", topk=3):
    print(f"{label}: {score:.2%}")
```

That script:
1. Restores the exact model architecture defined in the notebook (`coatnet_0_rw_224.sw_in1k`).
2. Reapplies the ImageNet normalization and 224×224 resolution expected by the pretrained backbone.
3. Returns the top‑K classes together with confidence scores, formatted with readable breed names.