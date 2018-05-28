#!/usr/bin/python
# -*- coding: UTF-8 -*-
import json
import argparse


def load_config():
    parser = argparse.ArgumentParser(description='astari ai')
    parser.add_argument("mode", choices=["game", "train"], help="play game or train")
    parser.add_argument("config", help="json config file path")
    args = parser.parse_args()
    with open(args.config, "r") as fp:
        config_map = json.load(fp)
        return args.mode, config_map["game"], config_map["train"]


run_mode, game_config, train_config = load_config()
