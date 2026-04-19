# SpiralRef: Hand Mesh Recovery with Modified ViT Backbone

This repository contains the implementation of **SpiralRef**, built on top of the original HaMeR framework.

SpiralRef modifies the Vision Transformer (ViT) backbone and related components to improve efficiency and representation learning, while preserving the original training and evaluation pipeline of HaMeR.

---

## Installation

Clone the repository (with submodules if using third-party dependencies):

```bash
git clone --recursive https://github.com/phuongttn/spiralref.git
cd spiralref
```
Create a virtual environment
```bash
python3.10 -m venv .spiralref
source .spiralref/bin/activate
```
or with conda: 
```bash
conda create --name spiralref python=3.10
conda activate spiralref
```
Install dependencies:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu117
pip install -e .[all]
pip install -v -e third-party/ViTPose
```
You also need to download the trained models:
```bash
bash fetch_demo_data.sh
```
## Training
First, download the training data to ./hamer_training_data/ by running:
```bash
bash fetch_training_data.sh
```
Then you can start training using the following command:
```bash
python train.py exp_name=spiralref data=mix_all experiment=hamer_vit_transformer trainer=gpu launcher=local
```
Checkpoints and logs will be saved to ./logs/.
