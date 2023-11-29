import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from .data import Caltech101DataModule
from .model import ViTLightningModule


def vit_train():
    dm = Caltech101DataModule()
    model = ViTLightningModule()

    wandb_log = WandbLogger(project="vit", name="vit_1", save_dir="./wandb")

    checkpoint = ModelCheckpoint(save_top_k=3, monitor="val acc")
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    trainer = L.Trainer(
        max_epochs=30,
        accelerator="gpu",
        devices=[0],
        logger=wandb_log,
        callbacks=[checkpoint, lr_monitor],
    )
    trainer.fit(model=model, datamodule=dm)


if __name__ == "__main__":
    L.seed_everything(1702)
    vit_train()
