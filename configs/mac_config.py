from dataclasses import dataclass

from configs.Config import (
    Config,
    GeneralConfig,
    ModelConfig,
    OptimizerConfig,
    TrainingConfig,
)


@dataclass
class MacConfig(Config):
    general: GeneralConfig
    model: ModelConfig
    optimizer: OptimizerConfig
    training: TrainingConfig
