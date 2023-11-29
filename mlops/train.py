import hydra
import lightning as L
from hydra.core.config_store import ConfigStore
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from .config import Params
from .data import EuroSATDataModule
from .model import ViTLightningModule

cs = ConfigStore.instance()
cs.store(name="params", node=Params)


@hydra.main(config_path="config", config_name="config", version_base="1.3")
def main(cfg: Params):
    L.seed_everything(cfg.training.seed)
    dm = EuroSATDataModule(data_dir=cfg.data.path)
    model = ViTLightningModule(num_classes=cfg.model.num_classes, lr=cfg.training.lr)

    wandb_log = WandbLogger(
        project=cfg.logging.project,
        name=cfg.logging.name,
        save_dir=cfg.logging.save_dir,
    )

    checkpoint = ModelCheckpoint(
        save_top_k=cfg.logging.save_top_k, monitor=cfg.logging.monitor
    )
    lr_monitor = LearningRateMonitor(logging_interval=cfg.logging.logging_interval)

    trainer = L.Trainer(
        max_epochs=cfg.training.epochs,
        accelerator="gpu",
        devices=[cfg.training.gpu_id],
        logger=wandb_log,
        callbacks=[checkpoint, lr_monitor],
    )
    trainer.fit(model=model, datamodule=dm)


if __name__ == "__main__":
    main()
