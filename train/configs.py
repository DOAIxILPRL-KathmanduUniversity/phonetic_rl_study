"""Default Training Configurations & Hyperparameters."""

PRETRAIN_CONFIG = {
    "arch": "mamba",            # 'mamba', 'cnn', 'transformer', 'lstm'
    "d_model": 256,
    "dim_feedforward": 1024,
    "num_layers": 4,            # 4 for mamba, 6 for cnn/transformer, 3 for lstm
    "nhead": 4,
    "batch_size": 128,
    "lr": 3e-4,
    "epochs": 5,
    "weight_decay": 0.01,
    "max_len": 22,
    "train_data": "data/train_split.csv",
    "val_data": "data/val_split.csv",
    "vocab_file": "data/vocab.json",
    "checkpoint_dir": "checkpoints"
}

RL_CONFIG = {
    "algorithm": "scst",        # 'scst', 'a2c', 'ppo', 'reinforce'
    "reward_type": "composite_balanced", # 'sparse_exact_match', 'dense_cer', 'composite_balanced', 'dominant_phonetic'
    "lr": 5e-5,
    "epochs": 3,
    "steps_per_epoch": 100,
    "batch_size": 64,
    "temperature": 1.0,
    "gamma": 0.99,
    "clip_epsilon": 0.2,       # For PPO
    "entropy_weight": 0.01,
    "checkpoint_dir": "checkpoints"
}
