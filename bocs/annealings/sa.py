import os
import hydra
import numpy as np
import numpy.typing as npt
from typing import Optional
from log import get_logger
from utils import sample_integer_matrix, encode_one_hot, decode_one_hot
from configs import BOCSConfig
from surrogates import BayesianLinearRegressor
from aquisitions import simulated_annealing
from exps import find_optimum, load_study
from dotenv import load_dotenv
from hydra.core.config_store import ConfigStore

cs = ConfigStore()
cs.store(name="bocs_config", node=BOCSConfig)
load_dotenv()
logger = get_logger(__name__)
cfg: Optional[BOCSConfig] = None


def bocs_sa_ohe(objective,
                low: int,
                high: int,
                n_vars: int,
                n_init: int = 10,
                n_trial: int = 100,
                n_add: int = 5):
    # Initial samples
    X = sample_integer_matrix(n_init, low, high, n_vars)
    y = objective(X)

    # Convert to one hot
    range_vars = high - low + 1
    X = encode_one_hot(low, high, n_vars, X)

    # Define surrogate model
    blr = BayesianLinearRegressor(range_vars * n_vars, 2)
    blr.fit(X, y)

    for i in range(n_trial):
        X_new = []
        qubo = blr.to_qubo()
        while len(X_new) < n_add:
            opt_X, _ = simulated_annealing(
                qubo,
                n_vars,
                range_vars,
                num_sweeps=1000)
            for j in range(len(opt_X)):
                if len(X_new) < n_add and np.sum(opt_X[j, :]) == n_vars:
                    X_new.append(opt_X[j, :])

        X_new = np.atleast_2d(X_new)
        y_new = objective(decode_one_hot(low, high, n_vars, X_new))
        logger.info(y_new)

        # Update posterior
        X = np.vstack((X, X_new))
        y = np.hstack((y, y_new))

        # Update surrogate model
        blr.fit(X, y)

        # log and save current solution
        logger.info(f"iteration {i}, current best: {np.max(y)}")

    X = X[n_init:, :]
    y = y[n_init:]
    return X, y


def bayesian_optimization(
        alpha: npt.NDArray,
        low: int,
        high: int):
    # define objective
    def objective(X: npt.NDArray) -> npt.NDArray:
        return alpha @ X.T

    # find global optima
    opt_x, opt_y = find_optimum(objective, low, high, len(alpha))
    logger.info(f'opt_y: {opt_y}, opt_x: {opt_x}')

    _, y = bocs_sa_ohe(objective,
                       low=low,
                       high=high,
                       n_vars=len(alpha))
    y = np.maximum.accumulate(y)

    return opt_y - y


@hydra.main(version_base="1.2",
            config_path="/root/bocs/configs",
            config_name="config")
def main(config: BOCSConfig):
    global cfg
    cfg = config

    # load study, extract
    study = load_study(cfg.base.exp, f'{cfg.base.n_vars}.json')
    alpha = study['alpha']
    logger.info(f'experiment: {cfg.base.exp}, n_vars: {cfg.base.n_vars}')

    res = bayesian_optimization(alpha[cfg.base.id], cfg.base.low, cfg.base.high)

    # save
    filepath = f"{cfg.project.runs}/annealings/sa/{cfg.base.exp}/{cfg.base.n_vars}/{cfg.base.id}_{cfg.base.low}{cfg.base.high}"
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    np.save(filepath, res)


if __name__ == "__main__":
    main()
