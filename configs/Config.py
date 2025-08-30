from dataclasses import dataclass
from typing import IO, Union
import dacite
import yaml


@dataclass
class GeneralConfig:

    seed: int


@dataclass
class ModelConfig:
    d_model: int
    n_heads: int
    seq_len: int
    n_persist: int
    n_mem_tokens: int
    mem_hidden: int
    chunk_size: int
    weight_decay_memory: float
    momentum_memory: float


@dataclass
class OptimizerConfig:
    lr_model: float
    lr_memory: float
    opt_learning_rate: float
    opt_weight_decay: float


@dataclass
class TrainingConfig:
    batch_size: int
    max_steps: int


@dataclass
class Config:

    @classmethod
    def from_yaml(cls, path: Union[str, IO]) -> "Config":
        """Load configuration from a YAML file."""
        if isinstance(path, str):
            with open(path, "r") as f:
                config_data = yaml.safe_load(f)
        else:
            config_data = yaml.safe_load(path)

        return dacite.from_dict(data_class=cls, data=config_data)
