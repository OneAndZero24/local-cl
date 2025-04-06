## Models
Full models and custom layers.
- `inc_classifier.py` - `IncerementalClassifier` automatically extends given layer to handle new classes in CIL, used as head in each model
- `cl_module_abc.py` - `CLModuleABC` acts as a base class for CL Modules that record activations
- `lenet.py` - LeNet
- `mlp.py` - Cutomizable MLP wrapper
- `layer/` - Custom layer implementations. Use `instantiate` & `instantiate2D` API
