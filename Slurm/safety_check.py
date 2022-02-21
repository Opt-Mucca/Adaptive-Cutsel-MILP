#! /usr/bin/env python
import os
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('outfile', type=str)
    args = parser.parse_args()

    assert not os.path.isfile(args.outfile)
    assert os.path.isdir(os.path.dirname(args.outfile))

    with open(args.outfile, 'w') as s:
        s.write('True')
