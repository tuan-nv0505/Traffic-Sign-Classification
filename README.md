<p align="center">
    <img src="images/architecture.png" width="300">
</p>


## Mamba For Vision: You Only Need 197k Parameters for Traffic Sign Classification


⭐If this work is helpful for you, please help star this repo. Thanks!🤗



<a name="Getting_Started">
</a>

## Getting Started

### Installation

**Step 1: Clone the VMamba repository:**

To get started, first clone the Traffic-Sign-Classification repository and navigate to the project directory:

```bash
git clone https://github.com/tuan-nv0505/Traffic-Sign-Classification.git
cd Traffic-Sign-Classification
```

**Step 2: Environment Setup:**

***Create and activate a new conda environment***

```bash
conda create -n vmamba
conda activate vmamba
```

***Install Dependencies***

```bash
pip install -r requirements.txt
```

**Step 3: Train:**


```bash
python train.py \
--epochs 70 \
--batch 64 \
--load_checkpoint \
--workers 4 \
--lr 1e-4 \
--path_data /your_path/GTSRB \
--trained /your_path/trained \
--logging /your_path/tensorboard \
--deep 4 \
--device cuda
