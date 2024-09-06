# PED: A Lightweight Prior-Encoding-Decoding Cascaded Depth Completion Framework for Robotic Grasping of Transparent Objects


## Requirements

The code has been tested under

- Ubuntu 20.04 + NVIDIA GeForce RTX 3060 (CUDA 11.1)
- PyTorch 1.9.0

### Experiments
---
The video of 6-Dof robotic experiments can be found at [this](https://youtu.be/wmkyNy8f5O0). 
---

### Dataset Preparation

- **TransCG** (recommended): See [TransCG Dataset](#transcg-dataset) section;
- **ClearGrasp** (syn and real): See [ClearGrasp official page](https://sites.google.com/view/cleargrasp);
- **Omniverse Object Dataset**: See [implicit-depth official repository](https://github.com/NVlabs/implicit_depth);


### Training

```
#Train on transcg dataset and test on transcg
python train.py --cfg ./configs/default.yaml

#Tran on CGsyn+ood and test on CGsyn and CGreal
python train.py --cfg ./configs/train_cgsyn+ood_val_cgsyn+cgreal.yaml

```


### Testing 

```
#Train on transcg dataset and test on transcg
python test.py --cfg ./configs/default.yaml

#Tran on CGsyn+ood and test on CGsyn and CGreal
python test.py --cfg ./configs/train_cgsyn+ood_val_cgsyn+cgreal.yaml

```


### Inference

```
#Train on transcg dataset and test on transcg
python sample_inference.py --cfg ./configs/inference.yaml

```

