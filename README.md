# Local Continual Learning
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

| Method | Status |
| ------ | -- |
| Naive | ✅ |
| LwF | ✅ |
| EWC | ✅ |
| Sharpening | ✅ |
| SI | ✅ |
| MAS | ✅ |
| RBFReg | ✅ |

| Custom Layers | Status |
| ------ | -- |
| Local | ✅ |
| RBF | ✅ |
| SingleRBFHead | ✅ |
| MultiRBFHead | ✅ |
| KAN | ❌ |
| LocalHead | ⭕️ |
| LocalConv2D | ⭕️ |

| Model | Status |
| ------ | -- |
| MLP | ✅ |
| LeNet | ⭕️ |

## Results
**SplitMNIST**
| Method | Normal | RBF | SingleRBFHead + Reg |
| ------ | ------ | --- | -------- |
| Naive | *19.97* | 19.95 | |
| LwF | ***23.23*** | NA | NA |
| EWC | 19.95 | | ***42.63*** |
| SI | 20.19 | ***32.28*** | 27.26 |
| MAS | 20.63 | 28.42 |  *35.2* |
| Sharpening | 19.95 | | |
| Regularization | NA | *19.88* | - |

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
