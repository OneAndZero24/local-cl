## Methods
Continual learning methods.
- `composer.py` - Main `Composer` class implements naive joint training flow, can be extended via plugins
- `method_plugin_abc.py` - `MethodABC` base class
- `regularization.py` - Loss components used in different methods for regularization
- `lwf.py` - Learning without Forgetting
- `ewc.py` - Elastic Weight Consolidation
- `mas.py` - Memory Aware Synapses
- `si.py` - Synaptic Intelligence
- `sharpening.py` - Sharpening https://cdn.aaai.org/Symposia/Spring/1993/SS-93-06/SS93-06-007.pdf
- `interval_penalization.py` - Adds regularization to preserve learned representations inside neuron-specific hypercubes and minimize activation variance, preventing drift when learning new tasks.
