import numpy as np
from exps import load_study, save_study
from itertools import product
from typing import Callable
import argparse
import numpy.typing as npt
from tqdm import tqdm
from utils import get_config

config = get_config()


def find_optimum(objective: Callable, low: int, high: int, n_vars: int, n_batch: int = 1):
    range_vars = high - low + 1
    assert range_vars ** n_vars < 2 ** 32, "The number of combinations for variables is too large."
    assert range_vars ** n_vars % n_batch == 0, "The number of combinations for variables must be divided by batch_size."

    # Generate all cases
    X = np.array(list(map(list, product(
        np.arange(low, high + 1).tolist(), repeat=n_vars)))).astype(np.float16)
    y = np.zeros(range_vars ** n_vars).astype(np.float16)

    # Split X into batches
    batches = np.split(X, n_batch, axis=0)
    batch_size = range_vars ** n_vars // n_batch
    for i, X_batch in enumerate(tqdm(batches)):
        y[i * batch_size: (i + 1) * batch_size] = objective(X_batch)

    # Find optimal solution
    max_idx = np.argmax(y)
    opt_x = X[max_idx, :].astype(np.float64)
    opt_y = y[max_idx].astype(np.float64)
    del y, batches

    return opt_x, opt_y


def find_bqp_optimum(args):
    n_vars = args.n_vars
    low = args.low
    high = args.high
    n_batch = args.n_batch

    # Extract study
    study = load_study('bqp', f'{n_vars}.json')
    Q = study['Q']
    Q = Q.astype(np.float16)
    lambda_l1 = study['lambda_l1']
    lambda_l2 = study['lambda_l2']

    # Define objective function
    def objective(X: npt.NDArray) -> npt.NDArray:
        return np.diag(X @ Q @ X.T) + \
            lambda_l1 * np.linalg.norm(X, ord=1, axis=1) + \
            lambda_l2 * np.linalg.norm(X, ord=2, axis=1)

    opt_x, opt_y = find_optimum(objective, low, high, n_vars, n_batch)

    optimum = {}
    optimum['opt_x'] = opt_x
    optimum['opt_y'] = opt_y
    study[f'{low}-{high}'] = optimum
    filepath = config['study_dir'] + 'bqp/' + f'{n_vars}.json'
    save_study(study, filepath)


def find_milp_optimum(args):
    n_vars = args.n_vars
    low = args.low
    high = args.high
    n_batch = args.n_batch

    # Extract study
    study = load_study('milp', f'{n_vars}.json')
    alpha = study['alpha']
    alpha = alpha.astype(np.float16)
    lambda_l1 = study['lambda_l1']
    lambda_l2 = study['lambda_l2']

    # Define objective function
    def objective(X: npt.NDArray) -> npt.NDArray:
        return alpha @ X.T + \
            lambda_l1 * np.linalg.norm(X, ord=1, axis=1) + \
            lambda_l2 * np.linalg.norm(X, ord=2, axis=1)

    opt_x, opt_y = find_optimum(objective, low, high, n_vars, n_batch)

    optimum = {}
    optimum['opt_x'] = opt_x
    optimum['opt_y'] = opt_y
    study[f'{low}-{high}'] = optimum
    filepath = config['study_dir'] + 'milp/' + f'{n_vars}.json'
    save_study(study, filepath)


def parse_args():
    parser = argparse.ArgumentParser(description='Find optimum for study.')
    subparsers = parser.add_subparsers()

    # Handler for Binary Quadratic Problem
    parser_bqp = subparsers.add_parser('bqp', help='see `bqp -h`')
    parser_bqp.add_argument('--n_vars', required=True, type=int)
    parser_bqp.add_argument('--low', required=False, default=0, type=int)
    parser_bqp.add_argument('--high', required=False, default=1, type=int)
    parser_bqp.add_argument('--n_batch', required=False, default=1, type=int)
    parser_bqp.set_defaults(handler=find_bqp_optimum)

    # # Handler for Mixed Integer Quadratic Problem
    # parser_bqp = subparsers.add_parser('miqp', help='see `miqp -h`')
    # parser_bqp.add_argument('--n_vars', required=True, type=int)
    # parser_bqp.add_argument('--low', required=False, default=0, type=int)
    # parser_bqp.add_argument('--high', required=False, default=1, type=int)
    # parser_bqp.add_argument('--n_batch', required=False, default=1, type=int)
    # parser_bqp.set_defaults(handler=find_miqp_optimum)

    # Handler for Mixed Integer Linear Problem
    parser_bqp = subparsers.add_parser('milp', help='see `milp -h`')
    parser_bqp.add_argument('--n_vars', required=True, type=int)
    parser_bqp.add_argument('--low', required=False, default=0, type=int)
    parser_bqp.add_argument('--high', required=False, default=1, type=int)
    parser_bqp.add_argument('--n_batch', required=False, default=1, type=int)
    parser_bqp.set_defaults(handler=find_milp_optimum)

    args = parser.parse_args()
    if hasattr(args, 'handler'):
        args.handler(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    parse_args()
