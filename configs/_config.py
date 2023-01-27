from dataclasses import dataclass

@dataclass
class BaseConfig:
    exp: str
    n_vars: int
    low: int
    high: int

@dataclass
class BOCSConfig:
    base: BaseConfig
    n_trial: int
    n_init: int
