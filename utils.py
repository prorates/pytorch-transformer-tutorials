#!/usr/bin/env python3

from typing import Any

import torch
import torchmetrics
import torchmetrics.text
from torch import nn
from torch.optim import Optimizer

from config import ConfigDict, get_best_model_params_path, get_weights_file_path, latest_weights_file_path


def collect_training_metrics(writer: Any, predicted: list[str], expected: list[str], global_step: int) -> None:
    if writer:
        # Evaluate the character error rate
        # Compute the char error rate
        cer_metric = torchmetrics.text.CharErrorRate()
        cer = cer_metric(predicted, expected)
        writer.add_scalar("validation cer", cer, global_step)
        writer.flush()

        # Compute the word error rate
        wer_metric = torchmetrics.text.WordErrorRate()
        wer = wer_metric(predicted, expected)
        writer.add_scalar("validation wer", wer, global_step)
        writer.flush()

        # Compute the BLEU metric
        bleu_metric = torchmetrics.text.BLEUScore()
        bleu = bleu_metric(predicted, expected)
        writer.add_scalar("validation BLEU", bleu, global_step)
        writer.flush()


def reload_model(
    config: ConfigDict, model: nn.Module, optimizer: Optimizer, initial_epoch: int, global_step: int
) -> tuple[nn.Module, int, Optimizer, int]:
    preload = config["preload"]
    model_filename = latest_weights_file_path(config) if preload == "latest" else get_weights_file_path(config, preload) if preload else None
    if model_filename:
        print(f"Preloading model {model_filename}")
        state = torch.load(model_filename)
        model.load_state_dict(state["model_state_dict"])  # JEB: This was not in the vide
        initial_epoch = state["epoch"] + 1
        optimizer.load_state_dict(state["optimizer_state_dict"])
        global_step = state["global_step"]
    else:
        print("No model to preload, starting from scratch")
    return model, initial_epoch, optimizer, global_step


def save_model(config: ConfigDict, model: nn.Module, optimizer: Optimizer, epoch: int, global_step: int, best_model_yet: bool = False) -> None:
    # Save the model at the end of every epoch
    model_filename = get_weights_file_path(config, f"{epoch:02d}")
    torch.save(
        {"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "global_step": global_step}, model_filename
    )

    if best_model_yet:
        best_model_filename = get_best_model_params_path(config)
        torch.save(
            {"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "global_step": global_step},
            best_model_filename,
        )


def load_trained_model(config: ConfigDict, model: nn.Module) -> nn.Module:
    preload = config["preload"]
    model_filename = latest_weights_file_path(config) if preload == "latest" else get_weights_file_path(config, preload) if preload else None
    print(f"Preloading model {model_filename}")
    if model_filename:
        state = torch.load(model_filename)
        model.load_state_dict(state["model_state_dict"])  # JEB: This was not in the video
    else:
        raise ValueError(f"{model_filename} Pretrained Model does not exist")
    return model
