import argparse
import logging
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import wandb

import src.tasks as tasks
from src.common import TrainConfig, setup_logger, CustomWandbCallback
from src.utils import load_yml

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Training with HuggingFace Trainer")

    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument('--wandb-key', type=str, required=False, help="weights & biases key.")

    args = parser.parse_args()

    return args


def main() -> None:
    # Parse arguments
    args = parse_args()

    # Setup logger
    setup_logger()

    # Load training config
    train_cfg = TrainConfig(**load_yml(args.cfg_path))

    # Set seeds for reproducibility
    if train_cfg.run_config.seed is not None:
        random.seed(train_cfg.run_config.seed)
        np.random.seed(train_cfg.run_config.seed)
        torch.manual_seed(train_cfg.run_config.seed)
        torch.cuda.manual_seed_all(train_cfg.run_config.seed)
        # Make cudnn deterministic for reproducibility
        cudnn.deterministic = True
        cudnn.benchmark = False
        logger.info(f"Set random seed to {train_cfg.run_config.seed}")

    # Setup file logging
    file_name = args.cfg_path.split('/')[-1].replace('.yml', '').replace('.yaml', '')
    log_dir = os.getenv("LOG_DIR", "./logs")
    os.makedirs(log_dir, exist_ok=True)

    from src.utils import now
    job_id = now()
    log_file = f'{log_dir}/{file_name}_{job_id}.log'

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info(f"Starting training with config: {args.cfg_path}")

    # Initialize wandb if key is provided
    if args.wandb_key:
        wandb.login(key=args.wandb_key)

    # Setup task and build trainer
    # The Trainer will handle distributed training internally using Accelerate
    task = tasks.setup_task(train_cfg)
    trainer = task.build_trainer()

    # Add custom callbacks
    if args.wandb_key:
        trainer.add_callback(CustomWandbCallback())

    # Train the model
    logger.info("Starting training...")
    trainer.train()

    # Save model
    logger.info("Training completed. Saving model...")
    trainer.save_model()

    if args.wandb_key:
        wandb.finish()

    logger.info("Training finished successfully!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        # Cleanup
        if wandb.run is not None:
            wandb.finish()
        raise
    except Exception as e:
        logger.exception(f"Training failed with error: {e}")
        # Cleanup
        if wandb.run is not None:
            wandb.finish()
        raise