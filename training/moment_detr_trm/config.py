from __future__ import annotations

import os
import shutil
from pathlib import Path
import yaml
from easydict import EasyDict

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = REPO_ROOT / "configs" / "moment_detr_trm"

class BaseOptionsTRM:
    """Load configuration stack for Moment-DETR-TRM."""

    def __init__(self, model: str = "moment_detr_trm", dataset: str = "charades_sta_semantic_novelty", feature: str = "clip_slowfast", resume: str | None = None):
        self.model = model
        self.dataset = dataset
        self.feature = feature
        self.resume = resume
        self.opt = {}

    @property
    def option(self):
        if not self.opt:
            raise RuntimeError("option is empty. Did you run parse()?")
        return self.opt

    def update(self, yaml_file: Path) -> None:
        with yaml_file.open("r", encoding="utf-8") as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
        if data:
            self.opt.update(data)

    def parse(self) -> None:
        cfgs = [
            CONFIG_ROOT / "base.yml",
            CONFIG_ROOT / "feature" / f"{self.feature}.yml",
            CONFIG_ROOT / "model" / f"{self.model}.yml",
            CONFIG_ROOT / "dataset" / f"{self.dataset}.yml",
        ]
        for cfg in cfgs:
            if not cfg.exists():
                raise FileNotFoundError(f"Missing config file: {cfg}")
            self.update(cfg)

        self.opt = EasyDict(self.opt)

        results_root = Path(self.opt.results_dir)
        if not results_root.is_absolute():
            results_root = REPO_ROOT / results_root

        if self.resume:
            results_dir = results_root / self.model / f"{self.dataset}_finetune" / self.feature
        else:
            results_dir = results_root / self.model / self.dataset / self.feature

        self.opt.results_dir = str(results_dir)
        self.opt.ckpt_filepath = str(results_dir / self.opt.ckpt_filename)
        self.opt.train_log_filepath = str(results_dir / self.opt.train_log_filename)
        self.opt.eval_log_filepath = str(results_dir / self.opt.eval_log_filename)

    def clean_and_makedirs(self, overwrite: bool = False) -> None:
        results_dir = Path(self.opt.results_dir)
        if results_dir.exists():
            if overwrite:
                shutil.rmtree(results_dir)
                results_dir.mkdir(parents=True, exist_ok=True)
        else:
            results_dir.mkdir(parents=True, exist_ok=True)
