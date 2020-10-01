#!/usr/bin/env python

import itertools

from force.workflow import cli, slurm


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--clear', action='store_true')
    parser.add_argument('-p', '--print', action='store_true')
    parser.add_argument('-c', '--cancel', type=int, default=None)
    parser.add_argument('-r', '--resubmit', type=str, default=None)
    args = parser.parse_args()

    if args.clear:
        slurm.clear()

    if args.print:
        slurm.update_metadata()
        slurm.print_metadata()

    if args.cancel is not None:
        slurm.cancel(args.cancel)

    if args.resubmit is not None:
        slurm.resubmit(args.resubmit)