## Models
Full models and custom layers.
- `inc_classifier.py` - `IncerementalClassifier` automatically extends given layer to handle new classes in CIL, used as head in each model
- `cl_module_abc.py` - `CLModuleABC` acts as a base class for CL Modules that record activations
- `lenet.py` - LeNet
- `mlp.py` - Customizable MLP wrapper
- `densenet.py` - Dense architecture - each layer receives concatenated outputs of all previous layers, head receives all outputs
- `big_model.py` - "Big model" - pretrained torchvision backbone with custom head
- `layer/` - Custom layer implementations. Use `instantiate` & `instantiate2D` API
