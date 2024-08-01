#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Returns comma-separated list of some files in a folder')
    parser.add_argument('path', type=str, help='dataset folder')
    parser.add_argument('--num-files', type=int, help='number of files to retrieve', default=None)

    args = parser.parse_args()

    if args.path is None or not os.path.isdir(args.path):
        raise ValueError('Invalid path {}.'.format(args.path))

    total_files = [os.path.join(args.path, f) for f in sorted(os.listdir(args.path)) if os.path.isfile(os.path.join(args.path, f))]

    if args.num_files is None:
        print(','.join(total_files))

    elif args.num_files > len(total_files) or args.num_files <= 0:
        raise ValueError('Invalid number of files {}'.format(args.num_files))
    
    else:
        selected_files = [total_files[int(i*((len(total_files)-1)/(args.num_files-1)))] for i in range(args.num_files)]
        print(','.join(selected_files))
    
    exit(0)