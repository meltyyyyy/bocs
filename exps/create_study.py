import argparse
import datetime
import json
import os
from exps import sbqp, bqp, milp, knapsack, miqp
from utils import NumpyEncoder, NumpyDecoder, get_config

config = get_config()


def save_study(study: dict, filepath: str):
    dirname = os.path.dirname(filepath)
    os.makedirs(dirname, exist_ok=True)

    with open(filepath, 'w') as f:
        json.dump(study, f, cls=NumpyEncoder, indent=2)


def load_study(exp: str, filename: str):
    filepath = config['study_dir'] + exp + '/' + filename
    with open(filepath, 'r') as f:
        study = json.load(f, cls=NumpyDecoder)

    return study


def create_knapsack(args):
    n_vars = args.n_vars
    n_runs = args.n_runs
    lambda_l1 = args.lambda_l1
    lambda_l2 = args.lambda_l2
    v, w, w_max = knapsack(n_vars)
    today = datetime.datetime.today()

    study = {
        'n_vars': n_vars,
        'n_runs': n_runs,
        'v': v,
        'w': w,
        'w_max': w_max,
        'lambda_l1': lambda_l1,
        'lambda_l2': lambda_l2,
        'created_at': today.strftime('%Y-%m-%d')
    }

    filepath = config['study_dir'] + 'knapsack/' + f'{n_vars}.json'

    save_study(study, filepath)


def create_sbqp(args):
    n_vars = args.n_vars
    n_runs = args.n_runs
    alpha = args.alpha
    lambda_l1 = args.lambda_l1
    lambda_l2 = args.lambda_l2
    Q = sbqp(n_vars, alpha)
    today = datetime.datetime.today()

    study = {
        'n_vars': n_vars,
        'n_runs': n_runs,
        'Q': Q,
        'lambda_l1': lambda_l1,
        'lambda_l2': lambda_l2,
        'created_at': today.strftime('%Y-%m-%d')
    }

    filepath = config['study_dir'] + 'sbqp/' + f'{n_vars}.json'

    save_study(study, filepath)


def create_bqp(args):
    n_vars = args.n_vars
    n_runs = args.n_runs
    lambda_l1 = args.lambda_l1
    lambda_l2 = args.lambda_l2
    Q = bqp(n_vars)
    today = datetime.datetime.today()

    study = {
        'n_vars': n_vars,
        'n_runs': n_runs,
        'Q': Q,
        'lambda_l1': lambda_l1,
        'lambda_l2': lambda_l2,
        'created_at': today.strftime('%Y-%m-%d')
    }

    filepath = config['study_dir'] + 'bqp/' + f'{n_vars}.json'

    save_study(study, filepath)


def create_miqp(args):
    n_vars = args.n_vars
    n_runs = args.n_runs
    lambda_l1 = args.lambda_l1
    lambda_l2 = args.lambda_l2
    today = datetime.datetime.today()

    study = {
        'n_vars': n_vars,
        'n_runs': n_runs,
        'Q': [],
        'lambda_l1': lambda_l1,
        'lambda_l2': lambda_l2,
        'created_at': today.strftime('%Y-%m-%d')
    }

    for _ in range(n_runs):
        Q = miqp(n_vars)
        study['Q'].append(Q)

    filepath = config['study_dir'] + 'miqp/' + f'{n_vars}.json'

    save_study(study, filepath)


def create_milp(args):
    n_vars = args.n_vars
    n_runs = args.n_runs
    lambda_l1 = args.lambda_l1
    lambda_l2 = args.lambda_l2
    today = datetime.datetime.today()

    study = {
        'n_vars': n_vars,
        'n_runs': n_runs,
        'alpha': [],
        'lambda_l1': lambda_l1,
        'lambda_l2': lambda_l2,
        'created_at': today.strftime('%Y-%m-%d')
    }

    for _ in range(n_runs):
        alpha = milp(n_vars)
        study['alpha'].append(alpha)

    filepath = config['study_dir'] + 'milp/' + f'{n_vars}.json'

    save_study(study, filepath)


def parse_args():
    parser = argparse.ArgumentParser(description='Create study for bocs.')
    subparsers = parser.add_subparsers()

    # Handler for knapsack
    parser_kns = subparsers.add_parser('knapsack', help='see `knapsack -h`')
    parser_kns.add_argument('--n_runs', required=False, type=int, default=50)
    parser_kns.add_argument('--n_vars', required=False, type=int, default=10)
    parser_kns.add_argument(
        '--lambda_l1', required=False, type=float, default=0)
    parser_kns.add_argument(
        '--lambda_l2', required=False, type=float, default=0)
    parser_kns.set_defaults(handler=create_knapsack)

    # Handler for Binary Quadratic Problem
    parser_bqp = subparsers.add_parser('bqp', help='see `bqp -h`')
    parser_bqp.add_argument('--n_runs', required=False, type=int, default=50)
    parser_bqp.add_argument('--n_vars', required=False, type=int, default=10)
    parser_bqp.add_argument(
        '--lambda_l1', required=False, type=float, default=0)
    parser_bqp.add_argument(
        '--lambda_l2', required=False, type=float, default=0)
    parser_bqp.set_defaults(handler=create_bqp)

    # Handler for Sparse Binary Quadratic Problem
    parser_sbqp = subparsers.add_parser('sbqp', help='see `sbqp -h`')
    parser_sbqp.add_argument('--n_runs', required=False, type=int, default=50)
    parser_sbqp.add_argument('--n_vars', required=False, type=int, default=10)
    parser_sbqp.add_argument('--alpha', required=False, type=int, default=0.1)
    parser_sbqp.add_argument(
        '--lambda_l1', required=False, type=float, default=0)
    parser_sbqp.add_argument(
        '--lambda_l2', required=False, type=float, default=0)
    parser_sbqp.set_defaults(handler=create_sbqp)

    # Handler for Mixed Integer Quadratic Problem
    parser_miqp = subparsers.add_parser('miqp', help='see `miqp -h`')
    parser_miqp.add_argument('--n_runs', required=False, type=int, default=50)
    parser_miqp.add_argument('--n_vars', required=False, type=int, default=10)
    parser_miqp.add_argument(
        '--lambda_l1', required=False, type=float, default=0)
    parser_miqp.add_argument(
        '--lambda_l2', required=False, type=float, default=0)
    parser_miqp.set_defaults(handler=create_miqp)

    # Handler for Mixed Integer Linear Problem
    parser_sbqp = subparsers.add_parser('milp', help='see `milp -h`')
    parser_sbqp.add_argument('--n_runs', required=False, type=int, default=50)
    parser_sbqp.add_argument('--n_vars', required=False, type=int, default=10)
    parser_sbqp.add_argument(
        '--lambda_l1', required=False, type=float, default=0)
    parser_sbqp.add_argument(
        '--lambda_l2', required=False, type=float, default=0)
    parser_sbqp.set_defaults(handler=create_milp)

    args = parser.parse_args()
    if hasattr(args, 'handler'):
        args.handler(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    parse_args()
