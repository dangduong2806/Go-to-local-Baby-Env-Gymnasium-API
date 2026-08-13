python train.py --num-updates 100 --rollout-steps 256

Collect 256 transitions
→ calculate GAE advantages and returns
→ run PPO optimization
→ repeat 100 times

python evaluate.py `
  --checkpoint runs/ppo_gotolocal/checkpoint_best.pt `
  --episodes 10 `
  --video-dir runs/ppo_gotolocal/videos `
  --video-episodes 3