from dataclasses import dataclass, field

@dataclass
class Data:
    path: str = "./data"

@dataclass
class Model:
    num_classes: int = 10

@dataclass
class Training:
    batch_size: int = 32
    num_workers: int = 4
    lr: float = 0.003
    epochs: int = 30
    gpu_id: int = 0
    seed: int = 1702

@dataclass
class Logging:
    project: str = 'vit'
    name: str = 'vit_1'
    save_dir: str = './wandb'
    monitor: str = 'val acc'
    save_top_k: int = 3
    logging_interval: str = 'epoch'

@dataclass
class Params:
    data: Data = field(default_factory=Data)
    model: Model = field(default_factory=Model)
    training: Training = field(default_factory=Training)
    logging: Logging = field(default_factory=Logging)