# main.py
import os
import argparse
import yaml

import sys
sys.path.append(os.path.abspath("LiteHrNet"))

from train_val_geospace import run_training, is_main_process, log

def load_cfg(path: str) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    # ensure sub-dicts exist so .get() calls in train_val.py are safe
    cfg.setdefault("data", {})
    cfg.setdefault("model", {})
    cfg.setdefault("train", {})
    cfg.setdefault("loss", {})
    cfg.setdefault("eval", {})
    return cfg

def pretty(cfg: dict) -> str:
    # nice YAML dump for readability
    return yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False)

def main():
    parser = argparse.ArgumentParser(description="Spacecraft pose training")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--print-config", action="store_true", help="Print merged config and exit")
    args = parser.parse_args()

    cfg = load_cfg(args.config)

    if args.print_config and is_main_process():
        log("--------- Effective Config ---------")
        log(pretty(cfg))
        return

    # Helpful banner on rank-0
    if is_main_process():
        local_rank = os.environ.get("LOCAL_RANK", "0")
        world_size = os.environ.get("WORLD_SIZE", "1")
        log(f"Launching training | LOCAL_RANK={local_rank} WORLD_SIZE={world_size}")
        log("--------- Config (summary) ---------")
        log(pretty(cfg).strip())

    run_training(cfg, logger=log)

if __name__ == "__main__":
    main()