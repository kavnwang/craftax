# Craftax-Classic PPO, single-host 8xH100

This is a minimal JAX/Flax PPO trainer for `Craftax-Classic-Symbolic-v1`.
It trains from scratch with `jax.pmap`: every GPU owns a shard of environments
and gradients are averaged across the `devices` axis.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -U "jax[cuda12]"
pip install -r requirements.txt
```

## Run on 8 H100s

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py
```

Useful overrides:

```bash
python train.py \
  --total-timesteps 1e9 \
  --num-envs-per-device 2048 \
  --num-steps 64 \
  --num-minibatches 16 \
  --update-epochs 4
```

## Smoke test on any device count

```bash
python train.py \
  --require-num-devices 0 \
  --total-timesteps 1048576 \
  --num-envs-per-device 128 \
  --num-steps 32 \
  --checkpoint-interval-updates 0
```

## Files

- `model.py`: only the actor-critic model.
- `config.py`: only hyperparameters.
- `train.py`: environment sharding, PPO rollout, GAE, multi-GPU gradient averaging, logging, and checkpoints.

Checkpoints save `params.msgpack` plus `metadata.json` under `runs/<run_name>/checkpoints/`.
Metrics are appended to `runs/<run_name>/metrics.csv`.
