## Models

- `layer/*` - This subfolder contains implementation of fully connected and conv2d layers with locality property. As well as `instantiate`, `instantiate2D` factory methods.
- `inc_classifier.py` - `IncerementalClassifier` automatically extends given layer to handle output in class incremental setting.
- `activation_recording_abc.py` - `ActivationRecordingModuleABC` acts as a base class for Modules which record their layer activity on each step.
- `lenet.py` - Configurable `LeNet` implementation.
- `mlp.py` - Configurable `MLP` configuration.
- `local_head.py` - `LocalHead` version of `IncrementalClassifier` with scaling for local layers.