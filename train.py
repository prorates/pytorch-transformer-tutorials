#!/usr/bin/env python3
import sys
import getopt

from config import get_config

from tutorial1 import train_model1
from tutorial2 import train_model2
from tutorial3 import train_model3
from tutorial4 import train_model4
from tutorial5 import train_model5
from tutorial6 import train_model6
from tutorial7 import train_model7
from tutorial8 import train_model8


def main(argv):
    config_filename = None
    model_folder = None
    try:
        opts, args = getopt.getopt(argv, "hc:m:", ["config=", "modelfolder="])
    except getopt.GetoptError:
        print("train.py -c <config_file> -m <model_folder>")
        sys.exit(2)
    for opt, arg in opts:
        if opt == "-h":
            print("train.py -c <config_file> -m <model_folder>")
            sys.exit()
        elif opt in ("-c", "--config"):
            config_filename = arg
        elif opt in ("-m", "--modelfolder"):
            model_folder = arg

    # warnings.filterwarnings('ignore')
    config = get_config(config_filename, model_folder)

    match config["alt_model"]:
        case "model1":
            train_model1(config)
        case "model2":
            train_model2(config)
        case "model3":
            train_model3(config)
        case "model4":
            train_model4(config)
        case "model5":
            train_model5(config)
        case "model6":
            train_model6(config)
        case "model7":
            train_model7(config)
        case "model8":
            train_model8(config)
        case _:
            train_model1(config)


if __name__ == "__main__":
    main(sys.argv[1:])
