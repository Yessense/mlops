import logging
import os
from pathlib import Path
from typing import Any
import hydra
import lightning as L
import git
from hydra.core.config_store import ConfigStore

from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger, MLFlowLogger
import torch

# from hydra.experimental.callback import Callback
# from omegaconf import DictConfig, open_dict

from .config import Params
from .data import EuroSATDataModule
from .model import ViTLightningModule


cs = ConfigStore.instance()
cs.store(name="params", node=Params)


# def get_git_sha():
#     repo = git.Repo(search_parent_directories=True)
#     sha = repo.head.object.hexsha
#     return sha


# class GitCallback(Callback):
#     def on_job_start(self, config: DictConfig, **kwargs: Any) -> None:
#         sha = get_git_sha()
#         output_dir = Path(config.hydra.runtime.output_dir) / Path(
#             config.hydra.output_subdir
#         )
#         commit_sha_file = Path(output_dir) / "sha.txt"
#         with open(commit_sha_file, "w") as f:
#             f.write(sha)


@hydra.main(config_path="../config", config_name="params", version_base="1.3")
def main(cfg: Params):
    L.seed_everything(cfg.training.seed)
    dm = EuroSATDataModule(data_dir=cfg.data.path)
    model = ViTLightningModule(
        num_classes=cfg.model.num_classes,
        lr=cfg.training.lr,
    )

    wandb_logger = WandbLogger(
        project=cfg.logging.project,
        name=cfg.logging.name,
        save_dir=cfg.logging.wandb_log_dir,
    )

    mlf_logger = MLFlowLogger(
        experiment_name=cfg.logging.name,
        tracking_uri=cfg.logging.ml_flow_uri,
    )

    checkpoint = ModelCheckpoint(
        save_top_k=cfg.logging.save_top_k,
        monitor=cfg.logging.monitor,
    )
    lr_monitor = LearningRateMonitor(
        logging_interval=cfg.logging.logging_interval,
    )


    trainer = L.Trainer(
        max_epochs=cfg.training.epochs,
        accelerator="gpu",
        devices=[cfg.training.gpu_id],
        logger=mlf_logger,
        callbacks=[checkpoint, lr_monitor],
    )
    trainer.fit(
        model=model,
        datamodule=dm,
    )

    input_sample = torch.randn((1, 3, 224, 224))
    model.to_onnx(
        file_path=cfg.training.save_path_onnx,
        input_sample=input_sample,
        export_params=True,
    )

if __name__ == "__main__":
    main()
