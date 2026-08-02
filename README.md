## Efficient token sparsification based on spiral refinement for ViT-based hand description

This repository contains the implementation of **SpiralRef**, built on top of the original HaMeR framework.

Vision Transformers (ViTs) have achieved remarkable performance in feature representation for hand pose estimation (HPE) and hand mesh reconstruction (HMR), yet their quadratic self-attention substantially incurs high complexity. An efficient sparsification would sharply decrease the computational cost for the transformer encoding. To this end, a deterministic token sparsification (named SpiralRef) is introduced to reduce the input tokens subject to two significant concepts as follows. First, a comprehensive analysis is presented to determine that the dominant regions of hand features are actually occupied by small portions in a given hand image, i.e., there are many tokens that have been redundant for the embedding process due to their non-hand information. To figure out which ones are trivial to be discarded before the embedding process, a novel spiral-based refinement is then proposed by considering the adjacent relationships of these input tokens in a locality-preserving spiral order. Unlike several learning-based token pruning techniques, SpiralRef requires no additional learnable parameters or auxiliary supervision, making it lightweight and easily integrated into a ViT-based architecture. Experimental results for HPE and HMR on benchmark datasets have demonstrated the prominent efficacy of our proposals. For instance, at a 40% token dropping rate, SpiralRef with Hamba reduces the computational cost by up to 40.5% GFLOPs, while simultaneously improving hand estimation performances on HO3D-v2, e.g., PA-MPJPE from 7.7 to 7.4 (i.e., 3.9%) and PA-MPVPE from 7.9 to 7.7 (i.e., 2.5%). Implementation codes are available at https://github.com/phuongttn/spiralref.
---

## Installation

Clone the repository (with submodules if using third-party dependencies):

```bash
git clone --recursive https://github.com/phuongttn/spiralref.git
cd spiralref
```
Create a virtual environment
```bash
python3.10 -m venv spiralref
source spiralref/bin/activate
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
## Download pretrained models: 
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
python train.py exp_name=spiralref data=mix_all experiment=spiralref_vit_transformer trainer=gpu launcher=local
```
Checkpoints and logs will be saved to ./logs/.

## Acknowledgements

SpiralRef is developed based on the HaMeR framework.

We sincerely thank Georgios Pavlakos and the HaMeR authors for making their codebase publicly available.

Original HaMeR repository:
https://github.com/geopavlakos/hamer

HaMeR is licensed under the MIT License.
