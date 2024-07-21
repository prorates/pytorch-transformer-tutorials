#!/usr/bin/env python3
import pandas as pd
from datasets import load_dataset


def remove_unicode(srcfilename: str = "en.txt", tgtfilename: str = "fr.txt"):

    try:
        in_file = open(srcfilename, "r")
        src_data = in_file.read()
        src_data = src_data.replace(b"\xc2\xa0".decode("utf-8"), " ")
        src_data = src_data.replace(b"\xe2\x80\x89".decode("utf-8"), " ")
        src_data = src_data.replace(b"\xe2\x80\xaf".decode("utf-8"), " ")
        in_file.close()

        out_file = open(srcfilename.replace(".txt", "2.txt"), "w")
        out_file.write(src_data)
        out_file.close()
    except:
        print("error: '" + srcfilename + "' file not found")
        quit()

    try:
        in_file = open(tgtfilename, "r")
        trg_data = in_file.read()
        trg_data = trg_data.replace(b"\xc2\xa0".decode("utf-8"), " ")
        trg_data = trg_data.replace(b"\xe2\x80\x89".decode("utf-8"), " ")
        trg_data = trg_data.replace(b"\xe2\x80\xaf".decode("utf-8"), " ")
        in_file.close()

        out_file = open(tgtfilename.replace(".txt", "2.txt"), "w")
        out_file.write(trg_data)
        out_file.close()
    except:
        print("error: '" + tgtfilename + "' file not found")
        quit()

    print("creating dataset and iterator... ")


def create_dataset(srcfilename: str = "en.txt", tgtfilename: str = "fr.txt", data_files: str = "dataset.csv", max_strlen: int = 512):

    try:
        src_data = open(srcfilename).read().strip().split("\n")
    except:
        print("error: '" + srcfilename + "' file not found")
        quit()

    try:
        trg_data = open(tgtfilename).read().strip().split("\n")
    except:
        print("error: '" + tgtfilename + "' file not found")
        quit()

    print("creating dataset and iterator... ")

    raw_data = {"en": [line for line in src_data], "fr": [line for line in trg_data]}
    df = pd.DataFrame(raw_data, columns=["en", "fr"])

    mask = (df["en"].str.count(" ") < max_strlen) & (df["fr"].str.count(" ") < max_strlen)
    df = df.loc[mask]

    df.to_csv(data_files, index=False, sep="|")


def verify_dataset(data_files: str = "dataset.csv"):
    local_csv_data = load_dataset("csv", data_files=data_files, sep="|", split="train")
    local_csv_data


if __name__ == "__main__":
    # remove_unicode()
    create_dataset()
    verify_dataset()
