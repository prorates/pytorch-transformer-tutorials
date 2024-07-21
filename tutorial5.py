from pathlib import Path


from config import get_device, get_model_folder, get_config
from dataset3 import get_ds3, get_testing_ds3
from model5 import Transformer5, build_transformer5

from utils import load_trained_model


def build_model5(config: dict, vocab_src_len: int, vocab_tgt_len: int) -> Transformer5:
    model = build_transformer5(
        vocab_src_len,
        vocab_tgt_len,
        config["seq_len"],
        config["seq_len"],
        d_model=config["d_model"],
        N=config["N"],
        h=config["h"],
        dropout=config["dropout"],
        d_ff=config["d_ff"],
    )
    return model


def train_model5(config: dict):
    device = get_device()

    model_folder = get_model_folder(config)
    Path(model_folder).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds3(config, model_folder)
    model = build_model5(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    raise RuntimeError("Training for model5 not implemented")


def translate5(config: dict, sentence: str):
    device = get_device()

    model_folder = get_model_folder(config)
    if not Path.exists(Path(model_folder)):
        raise ValueError(f"{model_folder} model_folder does not exist")

    sentence, label, to5enizer_src, tokenizer_tgt = get_testing_ds3(config, model_folder, sentence)
    model = build_model5(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    # Load the pretrained weights
    model = load_trained_model(config, model)

    # if the sentence is a number use it as an index to the test set
    # run_translation(label, sentence, model, tokenizer_src, tokenizer_tgt, config['seq_len'], device)


def debug_code_model5(config: dict, device):
    config["model"] = "model5"
    config["datasource"] = "translate"
    config["lang_src"] = "en"
    config["lang_tgt"] = "fr"

    model_folder = get_model_folder(config)
    Path(model_folder).mkdir(parents=True, exist_ok=True)

    # train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds2(config, model_folder)
    model = build_model5(config, 500, 500).to(device)

    print(model)
    model.train()


if __name__ == "__main__":
    # warnings.filterwarnings('ignore')
    config = get_config()
    device = get_device()
    debug_code_model5(config, device)
