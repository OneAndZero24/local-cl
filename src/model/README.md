## Models
Full models and custom layers.
- `inc_classifier.py` - `IncrementalClassifier` automatically extends given layer to handle new classes in CIL, used as head in each model
- `regression_head.py` - `RegressionHead` for regression tasks, outputs continuous values
- `cl_module_abc.py` - `CLModuleABC` acts as a base class for CL Modules that record activations
- `lenet.py` - LeNet
- `mlp.py` - Customizable MLP wrapper (supports both classification and regression)
- `densenet.py` - Dense architecture - each layer receives concatenated outputs of all previous layers, head receives all outputs
- `big_model.py` - "Big model" - pretrained torchvision backbone with custom head
- `resnet18_interval.py` - ResNet18 with interval-based training, used for CIL
- `layer/` - Custom layer implementations. Use `instantiate` & `instantiate2D` API
