# IntervalNet

Code repository for "Continual Learning with Guarantees via Weight Interval Constraints" accepted to ICML 2022. 
Paper link: [https://arxiv.org/abs/2206.07996](https://arxiv.org/abs/2206.07996)

## Setup

Build the singularity image if needed:
`TMPDIR=~/tmp/singularity_tmpdir singularity build --fakeroot image.sif image.def`
(You may need to follow [this](https://sylabs.io/guides/3.5/admin-guide/user_namespace.html#fakeroot-feature) to be able to do this locally without root)

Create a [W&B](https://wandb.ai/) account and add the following content to your `~/.bashrc`:
`export WANDB_API_KEY="<YOUR_KEY>"`

Populate `.env` file with settings from `.env.example`:
```
cp .env.example .env
edit .env
```
Ensure the path to the singularity image file is correct.

## Training

Scripts for running the experiments from the paper are in the `scripts` directory. 


## Development

Main interval training logic is in `intervalnet/strategy.py`. Model specifics are in `intervalnet/models/interval.py`.
