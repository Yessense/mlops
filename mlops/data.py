import lightning as L
import torch
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from dvc.api import DVCFileSystem
from torchvision.datasets import EuroSAT


class EuroSATDataModule(L.LightningDataModule):
    def __init__(
        self, data_dir: str = "./data", batch_size: int = 32, num_workers: int = 4
    ) -> None:
        super().__init__()

        self.num_workers = num_workers
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.num_classes = 101
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(224, antialias=True),
                transforms.CenterCrop(224),
            ]
        )

    def prepare_data(self) -> None:
        fs = DVCFileSystem('.', subrepos=True, rev='master')
        fs.get("data", "data", recursive=True)

        # EuroSAT(
        #     root=self.data_dir,
        #     download=True,
        # )

    def setup(self, stage: str) -> None:
        generator = torch.Generator().manual_seed(0)
        if stage == "fit" or stage is None:
            eurosat_full = EuroSAT(self.data_dir, transform=self.transform)

            self.eurosat_train, self.eurosat_val, _ = random_split(
                eurosat_full,
                [0.6, 0.2, 0.2],
                generator=generator,
            )

        if stage == "test" or stage is None:
            _, _, self.eurosat_test = random_split(
                eurosat_full,
                [0.6, 0.2, 0.2],
                generator=generator,
            )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        dl = DataLoader(
            self.eurosat_train,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
        print(f"Len of train dataloader: {len(dl)}")
        return dl

    def val_dataloader(self) -> EVAL_DATALOADERS:
        dl = DataLoader(
            self.eurosat_val,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
        print(len(dl))
        print(f"Len of val dataloader: {len(dl)}")
        return dl

    def test_dataloader(self) -> EVAL_DATALOADERS:
        dl = DataLoader(
            self.eurosat_test,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
        print(f"Len of test dataloader: {len(dl)}")
        return dl
