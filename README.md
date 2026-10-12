# KPST

Experimental code for Class-Incremental Unsupervised Domain Adaptation (CIUDA) in hyperspectral image classification. The project supports the Pavia and Houston datasets through unified training and evaluation entry points while retaining the network architecture and default hyperparameters for each dataset.

The following four cross-domain tasks are supported:

| Dataset | Domain identifiers | Tasks | Spectral bands | Classes |
| --- | --- | --- | --- | --- |
| Pavia | `PC`: Pavia Centre; `UP`: University of Pavia | PC → UP, UP → PC | 102 | 7 |
| Houston | `H13`: Houston 2013; `H18`: Houston 2018 | H13 → H18, H18 → H13 | 48 | 7 |



## Environment and Installation


Use the following versions as a reference when setting up the environment:

| Dependency | Verified version |
| --- | --- |
| Python | 3.13.13 |
| PyTorch | 2.11.0 |
| NumPy | 2.4.4 |
| SciPy | 1.17.1 |
| h5py | 3.16.0 |
| scikit-learn | 1.8.0 |
| tqdm | 4.67.3 |

### Install Dependencies

Run the following commands from the project root:

```bash
conda create -n kpst python=3.13 -y
conda activate kpst

python -m pip install torch
python -m pip install numpy scipy h5py scikit-learn tqdm
```

When using an NVIDIA GPU, choose the PyTorch installation command for your operating system and compute platform from the [official PyTorch installation page](https://pytorch.org/get-started/locally/). The code uses the GPU when CUDA is available and falls back to the CPU otherwise.

`POT` is an optional dependency for unbalanced optimal transport. If it is not installed, the code uses the built-in PyTorch implementation:

```bash
python -m pip install POT
```

## Data Preparation

The code reads preprocessed `.mat` files. Both datasets must use a consistent seven-class labeling scheme and have the number of spectral bands listed above. Convert the raw data to the following format before running the code.

Default directory structure:

```text
KPST/
├── source_train.py
├── target_adapt.py
├── eval.py
├── auto.sh
├── tasks.sh
├── ...
├── Pavia/
│   ├── paviaC.mat
│   ├── paviaC_7gt.mat
│   ├── paviaU.mat
│   └── paviaU_7gt.mat
└── Houston/
    ├── Houston13.mat
    ├── Houston13_7gt.mat
    ├── Houston18.mat
    └── Houston18_7gt.mat
```

Data files contain the key `ori_data` with shape `[H, W, C]` in MATLAB. Label files contain the key `map` with shape `[H, W]`. Label `0` denotes the background, and labels `1–7` denote valid classes, which are converted to `0–6` after loading.

Pavia files are read using `scipy.io.loadmat`. Houston files must use the **MATLAB v7.3/HDF5** format; the code reads them with h5py and restores the MATLAB array orientation. The loader applies min-max normalization to each spectral band and extracts spatial patches using reflection padding.

Data can also be stored elsewhere: specify the location with `--data_path` when using the Python entry points, or with the environment variables described below when using the shell scripts.

## Quick Start

### Single Task

After preparing the data and activating the Python environment, run the following command from the project root:

```bash
bash auto.sh
```

By default, the script runs **PC → UP** with random seed **25** and GPU index **0**. It runs `source_train.py`, `target_adapt.py`, and `eval.py --eval_part both` in sequence.

Specify the direction and random seed:

```bash
bash auto.sh PC UP 25
bash auto.sh UP PC 50
bash auto.sh H13 H18 75
bash auto.sh H18 H13 100
```

Specify the GPU, Python executable, or data location:

```bash
GPU=1 PYTHON=python3 PAVIA_DATA_PATH=/path/to/Pavia bash auto.sh PC UP 25
GPU=0 HOUSTON_DATA_PATH=/path/to/Houston bash auto.sh H18 H13 25
```

### Batch Experiments

```bash
bash tasks.sh
```

Specify the locations of both datasets:

```bash
PAVIA_DATA_PATH=/path/to/Pavia \
HOUSTON_DATA_PATH=/path/to/Houston \
bash tasks.sh
```


## Run Each Step Separately

Run the Python entry points directly to tune parameters for individual steps.

```bash
python source_train.py \
  --dataset pavia --data_path ./Pavia \
  --source_domain PC --seed 25 --gpu 0 \
  --save_path ./results/pavia/PC_to_UP/seed_25

python target_adapt.py \
  --dataset pavia --data_path ./Pavia \
  --source_domain PC --target_domain UP --seed 25 --gpu 0 \
  --model_path ./results/pavia/PC_to_UP/seed_25 \
  --save_path ./results/pavia/PC_to_UP/seed_25

python eval.py \
  --dataset pavia --data_path ./Pavia \
  --source_domain PC --target_domain UP --seed 25 --gpu 0 \
  --save_path ./results/pavia/PC_to_UP/seed_25 --eval_part both
```


When running the Python entry points directly, the default output directory is `./results/pavia` or `./results/houston`. These directories do not automatically separate results by direction and seed. For multiple experiments, specify a separate `--save_path` for each experiment or use the shell scripts.

The shell scripts use the same seed and GPU across all steps of each experiment. When running steps separately, pass these values explicitly because the original defaults in the Python entry points vary by dataset and step. See [`datasets.py`](datasets.py) for more dataset-specific defaults.
