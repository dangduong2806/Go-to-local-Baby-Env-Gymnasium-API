# Stochastic sampling
python train.py `
  --num-updates 100 `
  --rollout-steps 512 `
  --validation-episodes 50 `
  --validation-interval 10 `
  --output-dir runs/ppo_gotolocal_validation

# Stochastic sampling
python evaluate.py `
  --checkpoint runs/ppo_gotolocal_validation/checkpoint_best.pt `
  --episodes 100 `
  --output runs/ppo_gotolocal_validation/evaluation_results.json `
  --video-dir runs/ppo_gotolocal_3actions/videos `
  --video-episodes 3
# Deterministic sampling
python evaluate.py `
  --checkpoint runs/ppo_gotolocal_3actions/checkpoint_best.pt `
  --episodes 100 `
  --deterministic `
  --output runs/ppo_gotolocal_3actions/evaluation_deterministic.json

This evaluates all 100 episodes, keeps the frames of the three best candidates

# Generate diagrams
python -m scripts.plot_results `
  --training-csv runs/ppo_gotolocal_3actions/training_metrics.csv `
  --evaluation-json runs/ppo_gotolocal_3actions/evaluation_results.json `
  --output-dir runs/ppo_gotolocal_3actions/plots `
  --smooth-window 10