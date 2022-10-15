import argparse


def create_knapsack(args):
    n_vars = args.n_vars
    n_runs = args.n_runs




def create_bqp(args):
    n_vars = args.n_vars
    n_runs = args.n_runs


def parse_args():
    parser = argparse.ArgumentParser(description='Create study for bocs.')
    subparsers = parser.add_subparsers()

    # Handler for knapsack
    parser_kns = subparsers.add_parser('knapsack', help='see `knapsack -h`')
    parser_kns.add_argument('--n_runs', required=False, type=int, default=50)
    parser_kns.add_argument('--n_vars', required=False, type=int, default=10)
    parser_kns.set_defaults(handler=create_knapsack)

    # Handler for bqp
    parser_bqp = subparsers.add_parser('bqp', help='see `bqp -h`')
    parser_bqp.add_argument('--n_runs', required=False, type=int, default=50)
    parser_bqp.add_argument('--n_vars', required=False, type=int, default=10)
    parser_bqp.add_argument('--alpha', required=False, type=int, default=0.1)
    parser_bqp.set_defaults(handler=create_bqp)

    args = parser.parse_args()
    if hasattr(args, 'handler'):
        args.handler(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    parse_args()
