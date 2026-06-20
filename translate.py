#!/usr/bin/env python3
import getopt
import sys

from config import get_config
from tutorial1 import translate1
from tutorial2 import translate2
from tutorial3 import translate3
from tutorial4 import translate4
from tutorial5 import translate5
from tutorial6 import translate6
from tutorial7 import translate7
from tutorial8 import translate8


def main(argv: list[str]) -> None:
    config_filename = None
    model_folder = None
    sentence = "I am not a very good a student."
    try:
        opts, _args = getopt.getopt(argv, "hc:m:s:", ["config=", "modelfolder=", "sentence="])
    except getopt.GetoptError:
        print("translate.py -c <config_file> -m <model_folder> -s <sentence>")
        sys.exit(2)
    for opt, arg in opts:
        if opt == "-h":
            print("translate.py -c <config_file> -m <model_folder> -s <sentence>")
            sys.exit()
        elif opt in ("-c", "--config"):
            config_filename = arg
        elif opt in ("-m", "--modelfolder"):
            model_folder = arg
        elif opt in ("-s", "--sentence"):
            sentence = arg

    # warnings.filterwarnings('ignore')
    config = get_config(config_filename, model_folder)

    match config["alt_model"]:
        case "model1":
            translate1(config, sentence)
        case "model2":
            translate2(config, sentence)
        case "model3":
            translate3(config, sentence)
        case "model4":
            translate4(config, sentence)
        case "model5":
            translate5(config, sentence)
        case "model6":
            translate6(config, sentence)
        case "model7":
            translate7(config, sentence)
        case "model8":
            translate8(config, sentence)
        case _:
            translate1(config, sentence)


if __name__ == "__main__":
    main(sys.argv[1:])
