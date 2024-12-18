mkdir /pip-build
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
TMPDIR=/pip-build pip --no-input --no-cache-dir install scikit-learn==1.0.2
TMPDIR=/pip-build pip --no-input --no-cache-dir install git+https://github.com/ContinualAI/avalanche.git@b295ebabda0c5fe76ca1ad0e046bd24fcaa41df4 git+https://github.com/karolpiczak/pytorch-yard.git@715f819ce0ad708b2aeeead47fa5e4d2e4431c3b git+https://github.com/pytorch/hydra-torch/#subdirectory=hydra-configs-torchvision
TMPDIR=/pip-build pip --no-input --no-cache-dir install coolname gdown ipywidgets opencv-python-headless python-dotenv rich torchmetrics tqdm typer
TMPDIR=/pip-build pip --no-input --no-cache-dir install wandb seaborn visdom
rm -rf /pip-build