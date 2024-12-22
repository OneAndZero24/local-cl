# Local Continual Learning
**Patryk Krukowski, Jan Miksa** @ *GMUM JU*

🚀 *Let's forget about catastrophic forgetting!* 🚀

## Setup
```
conda create -n "lcl" python=3.9
pip install -r requirements.txt
cp example.env .env
edit .env
```

## Launching Experiments
```
conda acitate lcl
WANDB_MODE={offline/online} HYDRA_FULL_ERROR={0/1} python src/main.py --config-name config 
```

## Project Structure
Based on 
[template](https://github.com/sobieskibj/templates/tree/master)
by Bartłomiej Sobieski