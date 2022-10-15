import argparse
import datetime
import json
import os
from knapsack import knapsack
from bqp import bqp
from utils import NumpyEncoder

STUDY_DIR = '/root/bocs/study/'


def save_as_json(study: dict, filepath: str):
    dirname = os.path.dirname(filepath)
    os.makedirs(dirname, exist_ok=True)

    with open(filepath, 'w') as f:
        json.dump(study, f, cls=NumpyEncoder, indent=2)


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

    filepath = STUDY_DIR + 'knapsack/' + f'{n_vars}.json'

    save_as_json(study, filepath)


def create_bqp(args):
    n_vars = args.n_vars
    n_runs = args.n_runs
    alpha = args.alpha
    lambda_l1 = args.lambda_l1
    lambda_l2 = args.lambda_l2
    Q = bqp(n_vars, alpha)
    today = datetime.datetime.today()

    study = {
        'n_vars': n_vars,
        'n_runs': n_runs,
        'Q': Q,
        'lambda_l1': lambda_l1,
        'lambda_l2': lambda_l2,
        'created_at': today.strftime('%Y-%m-%d')
    }

    filepath = STUDY_DIR + 'bqp/' + f'{n_vars}.json'

    save_as_json(study, filepath)


def parse_args():
    parser = argparse.ArgumentParser(description='Create study for bocs.')
    subparsers = parser.add_subparsers()

    # Handler for knapsack
    parser_kns = subparsers.add_parser('knapsack', help='see `knapsack -h`')
    parser_kns.add_argument('--n_runs', required=False, type=int, default=50)
    parser_kns.add_argument('--n_vars', required=False, type=int, default=10)
    parser_kns.add_argument('--lambda_l1', required=False, type=float, default=0)
    parser_kns.add_argument('--lambda_l2', required=False, type=float, default=0)
    parser_kns.set_defaults(handler=create_knapsack)

    # Handler for bqp
    parser_bqp = subparsers.add_parser('bqp', help='see `bqp -h`')
    parser_bqp.add_argument('--n_runs', required=False, type=int, default=50)
    parser_bqp.add_argument('--n_vars', required=False, type=int, default=10)
    parser_bqp.add_argument('--alpha', required=False, type=int, default=0.1)
    parser_bqp.add_argument('--lambda_l1', required=False, type=float, default=0)
    parser_bqp.add_argument('--lambda_l2', required=False, type=float, default=0)
    parser_bqp.set_defaults(handler=create_bqp)

    args = parser.parse_args()
    if hasattr(args, 'handler'):
        args.handler(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    parse_args()
