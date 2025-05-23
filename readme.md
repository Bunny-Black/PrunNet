# Learning Compatible Multi-Prize Subnetworks for Asymmetric Retrieval

### Environments

```bash
conda create -n bct python=3.7
conda activate bct
conda install faiss-cpu
pip install -r requirements.txt
```
### Dataset
#### GLDv2 
The training data for our project comes from a cleaned version of the original Google Landmarks Dataset v2 (GLDv2)(https://github.com/cvdfoundation/google-landmark), and we use 30% data for training. The whole training set in lmdb format will be released soon.
#### In-shop
The In-shop dataset can be obtained at (https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/InShopRetrieval.html).

### Quick Start

Configuration File
Modify the dataset path in configs/finetune/sub_model/gldv2.yaml.


Training
```bash
bash ./orders/sub_model_gldv2.sh  # gldv2
```
### Model Weight
The model weights will be released soon.
### Settings Explanation

COMP_LOSS.Type
Specifies the compatibility learning method.

'proj_with_cosine_sim': Our proposed method.

PrunNet:
Control the number of sub-models via SUB_MODEL.SPARSITY:
Sparsity indicates parameter sparsity but inversely correlates with model capacity.
Example: SUB_MODEL.SPARSITY=[0.2] → Sub-model retains 80% of parameters.

SwitchNet:
Enable SwitchNet by setting SUB_MODEL.USE_SWITCHNET.
Configure sub-model capacity via SNET.WIDTH_MULT_LIST:
Sub-model parameters scale quadratically (e.g., SNET.WIDTH_MULT_LIST= [0.25, 1.0] → 1/16 parameters at 0.25 width).

### Sub-model Implementations
PrunNet:

Source Code:
src/models/subnetworks/subnet.py: Layer implementations (convolution and fully-connected).
src/models/subnetworks/resnet18.py: ResNet18 implementation.
src/models/subnetworks/resnet50.py: ResNet50 implementation.
SwitchNet:

Source Code:
Slimmable layers: src/models/subnetworks/slimmable_ops.py
Model: src/models/subnetworks/switchnet.py (ResNet-based).

### Sub-model Computation Mechanism during Iteration
At each iteration:
Create sub-models via deepcopy from the parent model.
Compute loss and backpropagate gradients for sub-models.
Compute loss and backpropagate gradients for the parent model
(using its classifier to maintain backward compatibility).
This design ensures:

Sub-model computations do not interfere with the parent model.
Compatibility through the parent model’s classifier.

### Citation
If you find this project useful in your research, please consider citing:

```bibtex
@article{PrunNet,
  title={Learning Compatible Multi-Prize Subnetworks for Asymmetric Retrieval},
  author={Sun, Yushuai and Zhou, Zikun and Jiang, Dongmei and Wang, Yaowei and Yu, Jun and Lu, Guangming and Pei, Wenjie},
}
```
