# A Lightweight Prior-Encoding-Decoding Cascade Framework for Robust Depth Completion in Robotic Grasping of Transparent Objects Using RGB-D Sensors
## The paper is currently under review and all code will be made public.

![图片4](https://github.com/user-attachments/assets/350f8faa-8e3f-441e-aa8c-150ace1c54bc)

## Requirements

The code has been tested under

- Ubuntu 20.04 + NVIDIA GeForce RTX 3060 (CUDA 11.1)
- PyTorch 1.9.0

### Experiments

---
The video of 6-Dof robotic experiments and object sorting experiments can be found at [this](https://www.youtube.com/watch?v=34z3rwaktbU). 

![图片1](https://github.com/user-attachments/assets/4adee08f-cf99-4963-9f76-7e3ced54fd7a)
![图片2](https://github.com/user-attachments/assets/f2173aba-d94b-4bb3-a249-7334e4a1f5b7)


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

