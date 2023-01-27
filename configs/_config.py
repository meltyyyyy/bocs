from dataclasses import dataclass

@dataclass
class BaseConfig:
    exp: str
    id: int
    n_vars: int
    low: int
    high: int


@dataclass
class ProjectConfig:
    runs: str


@dataclass
class BOCSConfig:
    base: BaseConfig
    project: ProjectConfig
    n_trial: int
    n_init: int
