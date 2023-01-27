import os
import hydra
import numpy as np
import numpy.typing as npt
from configs import BOCSConfig
from log import get_logger
from utils import sample_integer_matrix, encode_one_hot, decode_one_hot
from surrogates import BayesianLinearRegressor
from exps import load_study, find_optimum
from dotenv import load_dotenv
from aquisitions import quantum_annealing
from dwave.system import DWaveSampler, EmbeddingComposite
from hydra.core.config_store import ConfigStore

cs = ConfigStore()
cs.store(name="bocs_config", node=BOCSConfig)
load_dotenv()
logger = get_logger(__name__)


def bocs_sqa_ohe(objective,
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

    # define sampler
    sampler = EmbeddingComposite(DWaveSampler(
        solver='DW_2000Q_6',
        token=os.environ['DWAVE_TOKEN'],
        endpoint=os.environ["DWAVE_API_ENDPOINT"]))

    for i in range(n_trial):
        X_new = []
        qubo = blr.to_qubo()
        while len(X_new) < n_add:
            opt_X, _ = quantum_annealing(
                qubo,
                sampler,
                n_vars,
                range_vars)

            for j in range(n_add):
                if len(X_new) < n_add and np.sum(opt_X[j, :]) == n_vars:
                    X_new.append(opt_X[j, :])

        # evaluate model objective at new evaluation point
        X_new = np.atleast_2d(X_new)
        y_new = objective(decode_one_hot(low, high, n_vars, X_new))

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

    _, y = bocs_sqa_ohe(objective,
                        low=low,
                        high=high,
                        n_vars=len(alpha))
    y = np.maximum.accumulate(y)

    return opt_y - y


@hydra.main(version_base="1.2",
            config_path="/root/bocs/configs",
            config_name="config")
def main(cfg: BOCSConfig):
    # load study, extract
    study = load_study(cfg.base.exp, f'{cfg.base.n_vars}.json')
    alpha = study['alpha']
    logger.info(f'experiment: {cfg.base.exp}, n_vars: {cfg.base.n_vars}')

    bayesian_optimization(alpha[cfg.base.id], cfg.base.low, cfg.base.high)


if __name__ == "__main__":
    main()
    # n_vars, low, high, i = int(sys.argv[1]), int(
    #     sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])

    # # run Bayesian Optimization with parallelization
    # def runner(i: int): return run_bayes_opt(alpha[i], low, high)
    # ans = runner(i)

    # filepath = config['output_dir'] + \
    #     f'annealings/sqa/{EXP}/qpu/{n_vars}/{i}_{low}{high}.npy'
    # os.makedirs(os.path.dirname(filepath), exist_ok=True)
    # np.save(filepath, ans)
