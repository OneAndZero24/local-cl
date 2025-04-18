# Local Continual Learning
<p align="right"><img style="float: right;" src="gmum.png" alt="logo" width="100"/></p>

**Patryk Krukowski, Jan Miksa** @ *GMUM JU*

🚀 *Let's forget about catastrophic forgetting!* 🚀

<p align="center"><img src="rbf.png" alt="rbf" width="200"/></p>

***Work in progress... There may be bugs and features might be missing.***

## Features
- Hydra Configuration
- WANDB Logging
- Lightning Fabric
- Custom Plugin System for Methods
- Incremental Classifier

| Method | Status | Custom Layers | Status | Model | Status | Scenario | Status | Dataset | Status |
| ------ | -- | ------ | -- | ------ | -- | ------ | -- | ------ | -- |
| Naive | ✅ | Local | ✅ | MLP | ✅ | CI | ✅ | MNIST | ✅ |
| LwF | ✅ | RBF | ✅ | LeNet | ✅ | DI | ✅ | ImageNet | ✅ |
| EWC | ✅ | SingleRBFHead | ✅ | | | TI | ✅ | CIFAR100 | ✅ |
| Sharpening | ✅ | MultiRBFHead | ✅ | | | II | ✅ | TinyImageNet | ✅ |
| SI | ✅ | KAN | ❌ | | | Permuted | ⭕️ |
| MAS | ✅ | LocalHead | ⭕️ |
| RBFReg | ✅ | LocalConv2D | ⭕️ |
| Dreaming | ✅ | TIMM Big Models | ✅ |
| Dynamic Loss Scaling | ✅ |

## Results
**SplitMNIST**
| Method | Full MLP | RBF+MultiRBFHead | RBF+SingleRBFHead | MLP+MultiRBFHead |
| ------ | -------- | ---------------- | ----------------- | ---------------- |
| Naive | 19.94 | 19.87 | *19.98* | 19.95 |
| LwF | *39.91* | **22.71** | NA | 25.38 |
| EWC | ***54.65*** | 19.74 | 21.86 | **51.81** |
| SI | *32.32* | 12.62 | 19.32 | 31.56 |
| MAS | 36.51 | 10.18 | **26.58** | *49.24* |
| Sharpening | 19.93 | *19.94* | 19.83 | 19.92 |
| NReg | NA | 19.90 | 19.85 | 19.92 |

## Commands
**Setup**
```
conda create -n "lcl" python=3.9
pip install -r requirements.txt
cp example.env .env
edit .env
```

**Launching Experiments**
```
conda activate lcl
WANDB_MODE={offline/online} HYDRA_FULL_ERROR={0/1} python src/main.py --config-name config 
```

## Acknowledgements
- Project Structure based on [template](https://github.com/sobieskibj/templates/tree/master) by Bartłomiej Sobieski
- PyTorchRBFLayer [repo](https://github.com/rssalessio/PytorchRBFLayer) by Alessio Russo
