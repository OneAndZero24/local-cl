## Models
Full models and custom layers.
- `inc_classifier.py` - `IncerementalClassifier` automatically extends given layer to handle new classes in CIL, used as head in each model
- `activation_recording_abc.py` - `ActivationRecordingModuleABC` acts as a base class for Modules that record activations
- `lenet.py` - **Deprecated** LeNet
- `mlp.py` - Cutomizable MLP wrapper
- `layer/` - Custom layer implementations. Use `instantiate` & `instantiate2D` API
