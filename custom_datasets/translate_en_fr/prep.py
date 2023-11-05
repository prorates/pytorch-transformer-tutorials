#!/usr/bin/env python3
import pandas as pd
from datasets import load_dataset


def create_dataset(srcfilename: str = "english.txt", tgtfilename: str = "french.txt", data_files: str = "dataset.csv", max_strlen: int = 512):

    try:
        src_data = open(srcfilename).read().strip().split('\n')
    except:
        print("error: '" + srcfilename + "' file not found")
        quit()

    try:
        trg_data = open(tgtfilename).read().strip().split('\n')
    except:
        print("error: '" + tgtfilename + "' file not found")
        quit()

    print("creating dataset and iterator... ")

    raw_data = {'en': [line for line in src_data], 'fr': [line for line in trg_data]}
    df = pd.DataFrame(raw_data, columns=["en", "fr"])

    mask = (df['en'].str.count(' ') < max_strlen) & (df['fr'].str.count(' ') < max_strlen)
    df = df.loc[mask]

    df.to_csv(data_files, index=False, sep="|")


def verify_dataset(data_files: str = "dataset.csv"):
    local_csv_data = load_dataset("csv", data_files=data_files, sep="|", split="train")
    local_csv_data


if __name__ == '__main__':
    create_dataset()
    verify_dataset()
