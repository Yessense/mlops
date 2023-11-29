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
    save_path_onnx: str = "./onnx/model.onnx"

@dataclass
class Logging:
    project: str = 'vit'
    name: str = 'vit_1'
    wandb_log_dir: str = "${hydra:run.dir}/wandb"
    ml_flow_uri: str = "file:${hydra:run.dir}/ml-runs" 

    monitor: str = 'val acc'
    save_top_k: int = 3
    logging_interval: str = 'epoch'

@dataclass
class GitInfo:
    _target_: str = "mlops.hydra_callbacks.GitInfo"

@dataclass
class GitLogging:
    git_logging: GitInfo = field(default_factory=GitInfo)

@dataclass
class Params:
    data: Data = field(default_factory=Data)
    model: Model = field(default_factory=Model)
    training: Training = field(default_factory=Training)
    logging: Logging = field(default_factory=Logging)
    callbacks: GitLogging = field(default_factory=GitLogging)
