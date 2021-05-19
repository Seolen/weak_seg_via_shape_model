

The implementation of [Weakly Supervised Volumetric Segmentation via Self-taught Shape Denoising Model](https://openreview.net/forum?id=Koyg3kvH-Mq)

# Installation
---
## Dependencies
- Python 3.6
- Pytorch 1.4
- Torchvision 0.4
- Cuda version 10.1

# Get Started
## Data Preparation
Please refer to [data_preparation.md]()
```
python data_process/process_pipeline2.py
```

## Training our model
```
# 1. train segmentation model with weak labels
python main.py --cfg exp/weak_trachea/1012_tra_r1_01.yaml --id 1012_tra_r1_01 --parallel

# 2. train shape denoising network
python main.py --cfg exp/ae/1013_tra_aelo_13.yaml --id 1013_tra_aelo_13 --parallel

# 3. iterative learning
python main.py --cfg exp/iterative/1016_trar1_emitw_24.yaml --id 1016_trar1_emitw_24 --parallel

```

## Inference with trained model
```
python main.py --cfg exp/iterative/1016_trar1_emitw_24.yaml --id 1016_trar1_emitw_24 --parallel \
  --demo 'val' --weight_path '/*/*/best_model.pth' [--ae_weight_path '/*/*/best_model.pth']
```

---
TODO
- [ ] Introduction of our shape-aware segmentation model
- [ ] Data preparation detail
- [ ] Trained model and final results





















