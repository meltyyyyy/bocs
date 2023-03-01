import os
import hydra
import gc
import numpy as np
import numpy.typing as npt
from typing import Optional
from log import get_logger
from utils import sample_integer_matrix, encode_one_hot, decode_one_hot
from configs import BOCSConfig
from surrogates import SparseBayesianLinearRegressor
from aquisitions import simulated_annealing
from exps import find_optimum, load_study
from dotenv import load_dotenv
from hydra.core.config_store import ConfigStore
from threadpoolctl import threadpool_limits

cs = ConfigStore()
cs.store(name="bocs_config", node=BOCSConfig)
load_dotenv()
logger = get_logger(__name__)
cfg: Optional[BOCSConfig] = None


def bocs_sblr_sa(objective,
                 low: int,
                 high: int,
                 n_vars: int,
                 n_init: int = 10,
                 n_trial: int = 750,
                 n_add: int = 1):
    reload_dir = f"{cfg.project.runs}/ohe/{cfg.base.exp}/sblr/{cfg.base.n_vars}/checkpoints/{cfg.base.id}"
    if os.path.exists(reload_dir):
        X, y = reload_data(reload_dir)
    else:
        # Initial samples
        X = sample_integer_matrix(n_init, low, high, n_vars)
        y = objective(X)

        # Convert to one hot
        X = encode_one_hot(low, high, n_vars, X)

    # Define surrogate model
    range_vars = high - low + 1
    sblr = SparseBayesianLinearRegressor(range_vars * n_vars, 2)
    with threadpool_limits(
            limits=int(os.environ['OPENBLAS_NUM_THREADS']),
            user_api='blas'):
        sblr.fit(X, y)

    for i in range((X.shape[0] - n_init) // n_add, n_trial):
        X_new = []
        qubo = sblr.to_qubo()
        while len(X_new) < n_add:
            opt_X, _ = simulated_annealing(
                qubo,
                n_vars,
                range_vars)
            for j in range(len(opt_X)):
                if len(X_new) < n_add and np.sum(opt_X[j, :]) == n_vars:
                    X_new.append(opt_X[j, :])

        X_new = np.atleast_2d(X_new)
        y_new = objective(decode_one_hot(low, high, n_vars, X_new))

        # Update posterior
        X = np.vstack((X, X_new))
        y = np.hstack((y, y_new))

        # Update surrogate model
        with threadpool_limits(
                limits=int(os.environ['OPENBLAS_NUM_THREADS']),
                user_api='blas'):
            sblr.fit(X, y)

        # log and save current solution
        logger.info(f"iteration {i}, current best: {np.max(y)}")

        # save checkpoint
        if (i + 1) % 10 == 0:
            save_checkpoint(X, y)
            logger.info(f"successfully saved iteration {i}, X.shape: {X.shape}, y.shape: {y.shape}")
        gc.collect()

    X = X[n_init:, :]
    y = y[n_init:]
    return X, y


def save_checkpoint(X: npt.NDArray, y: npt.NDArray):
    filepath = f"{cfg.project.runs}/ohe/{cfg.base.exp}/sblr/{cfg.base.n_vars}/checkpoints/{cfg.base.id}/"
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    np.save(filepath + f"X_{cfg.base.low}{cfg.base.high}.npy", X)
    np.save(filepath + f"y_{cfg.base.low}{cfg.base.high}.npy", y)


def reload_data(reload_dir: str):
    X = np.load(reload_dir + f"/X_{cfg.base.low}{cfg.base.high}.npy")
    y = np.load(reload_dir + f"/y_{cfg.base.low}{cfg.base.high}.npy")
    logger.info(f"reloading data X.shape: {X.shape}, y.shape: {y.shape}")
    return X, y


def bayesian_optimization(
        alpha: npt.NDArray,
        low: int,
        high: int):
    # define objective
    def objective(X: npt.NDArray) -> npt.NDArray:
        return alpha @ X.T

    _, y = bocs_sblr_sa(objective,
                        low=low,
                        high=high,
                        n_vars=len(alpha))
    y = np.maximum.accumulate(y)

    # find global optima
    opt_x, opt_y = find_optimum(objective, low, high, len(alpha), n_samples=int(10e6), is_heuristic=True)
    logger.info(f'opt_y: {opt_y}, opt_x: {opt_x}')

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
    filepath = f"{cfg.project.runs}/ohe/{cfg.base.exp}/sblr/{cfg.base.n_vars}/{cfg.base.id}_{cfg.base.low}{cfg.base.high}"
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    np.save(filepath, res)


if __name__ == "__main__":
    main()
