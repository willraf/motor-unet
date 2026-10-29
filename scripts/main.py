""" 
Script main

Main entry point for the full training/evaluation/prediction pipeline.

Usage:
    % python -m scripts.main --config configs/main.yaml
    % python -m scripts.main --config configs/main.yaml --default configs/default.yaml

Authors:
    Will Raftery

Date:
    31/07/2025
"""

import argparse
import logging

from omegaconf import OmegaConf
from pathlib import Path

from src.utils.utils import load_config, get_experiment_directory, set_seed

from scripts.split_muaps import main as split_muaps
from scripts.train import main as train
from scripts.eval import main as evaluate
from scripts.predict import main as predict

def main(cfg=None, overwrite=False):
    # setup basic logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler()
        ]
    )

    experiment_dir = get_experiment_directory(cfg)
    OmegaConf.save(cfg, experiment_dir / "config.yaml")

    # Split muaps if not already done
    muap_path = Path(cfg.data.muap_pkl_path)
    train_muap_path = Path(cfg.training.muap_pkl_path)
    eval_muap_path = Path(cfg.evaluation.muap_pkl_path)
    if not train_muap_path.exists() or not eval_muap_path.exists():
        logging.info("Training or evaluation muap files not found. Splitting muaps...")
        split_muaps(muaps_path=muap_path, train_muaps_path=train_muap_path, test_muaps_path=eval_muap_path)

    train_results = train(cfg, overwrite=overwrite)
    logging.info(f"Training completed with results: {train_results}")

    eval_results = evaluate(cfg, overwrite=overwrite)
    logging.info(f"Evaluation completed with results: {eval_results}")

    infer_results = predict(cfg, overwrite=overwrite)
    logging.info(f"Inference completed with results: {infer_results}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full training and evaluation pipeline")
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--default', type=str, required=False, default="configs/default.yaml")
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    cfg = load_config(args.config, default_path=args.default)

    random_seed = cfg.experiment.random_seed
    set_seed(random_seed)

    main(cfg, overwrite=args.overwrite)
