# neurl

## Setup

```bash
source hehehehaw/bin/activate
pip install "gymnasium[box2d]" torch matplotlib pillow
```

## Usage

```bash
# REINFORCE on CartPole; left/right swap at SWITCH
python reinforce_cartpole.py

# REINFORCE on LunarLander; left/right engine swap at SWITCH
python reinforce_lunar.py

# render a GIF from the reinforce_lunar.py checkpoints
python animate_lunar.py

# reinforce_lunar.py + at the switch, reset policy entropy to the untrained level
python reinforce_lunar_reset.py [--seed 0]              # with reset
python reinforce_lunar_reset.py [--seed 0] --no-reset   # same code, no reset

# phase 1: train trunk from scratch, 4x4 action-connect layer frozen at identity
# switch:  swap left/right engines, check the permutation matrix restores the policy exactly
# phase 2: train only the action-connect layer (--phase2 connect) or the whole network (--phase2 all)
python remap_lunar.py [--phase2 connect|all] [--reset-entropy] [--switch 3000] [--post-episodes 5000] \
                      [--lr 1e-3] [--connect-lr 1e-2] [--trunk-lr 1e-3] [--seed 0] [--eval-episodes 20]

# entropy, gradient norm, and return under random weight perturbations, on swapped controls
python probe_plateau.py <ckpt in checkpoints/remap_lunar/> [...] [--directions 8] [--episodes 10]
```

Hyperparameters for `reinforce_*.py` (`EPISODES`, `SWITCH`, `LR_*`, ...) are constants at the top of each file.

## Outputs

| Script | Writes |
|---|---|
| `reinforce_cartpole.py` | `checkpoints/cartpole_ckpt_{init,switch}.pt`, `checkpoints/reinforce_cartpole.pt`, `figures/cartpole_figure.png` |
| `reinforce_lunar.py` | `ckpt_{init,switch}.pt`, `reinforce_lunar.pt`, `figure.png` (in the working directory) |
| `reinforce_lunar_reset.py` | `checkpoints/lunar_reset/{reset,noreset}_seed<seed>_{init,switch,final}.pt`, `figures/lunar_reset/{reset,noreset}_seed<seed>.png` |
| `animate_lunar.py` | `figures/lunar.gif` |
| `remap_lunar.py` | `checkpoints/remap_lunar/<tag>_{switch,final}.pt`, `figures/remap_lunar/<tag>.png`; `<tag>` = `seed<seed>[_all_trunklr<lr>][_entreset]` |
| `probe_plateau.py` | prints only |
