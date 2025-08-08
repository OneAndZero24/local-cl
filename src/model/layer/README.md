## Layers
Custom layers implementations
- `local_module.py` - wrapper around `nn.Module` marking locality property PLEASE USE THIS
- `rbf.py` - Radial Basis Functions from PyTorchRBFLayer by Alessio Russo, with masked input feature dimensions.
- `local.py` - Local ReLU-based FC Layer
- `local_conv.py` - **Deprecated** Local ReLU-based Conv Layer
- `rbf_head.py` - Single RBF Head Layer
- `interval_activation.py` - Bounding activations per element inside estimated interval