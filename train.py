import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pytorch_lightning as pl
import torch
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import TQDMProgressBar
from torchvision.datasets import ImageFolder

from dataset import IntelImgClfDataModule
from model import LitResnet


sm_output_dir = Path(os.environ.get("SM_OUTPUT_DIR"))
sm_model_dir = Path(os.environ.get("SM_MODEL_DIR"))
num_cpus = int(os.environ.get("SM_NUM_CPUS"))

train_channel = os.environ.get("SM_CHANNEL_TRAIN")
test_channel = os.environ.get("SM_CHANNEL_TEST")

ml_root = Path("/opt/ml")


def get_training_env():
    sm_training_env = os.environ.get("SM_TRAINING_ENV")
    sm_training_env = json.loads(sm_training_env)

    return sm_training_env


def train(model, datamodule, sm_training_env):
    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=ml_root / "output" / "tensorboard" / sm_training_env["job_name"]
    )

    trainer = pl.Trainer(
        max_epochs=5,
        accelerator="auto",
        logger=[tb_logger],
        callbacks=[TQDMProgressBar(refresh_rate=5)],
    )
    trainer.fit(model, datamodule)

    return trainer


def save_scripted_model(model, output_dir):
    script = model.to_torchscript()
    # save for use in production environment
    torch.jit.save(script, output_dir / "model.scripted.pt")


def save_last_ckpt(trainer, output_dir):
    trainer.save_checkpoint(output_dir / "last.ckpt")


if __name__ == "__main__":

    img_dset = ImageFolder(train_channel)

    print(":: Classnames: ", img_dset.classes)

    datamodule = IntelImgClfDataModule(data_dir="", num_workers=num_cpus)
    datamodule.setup()

    model = LitResnet(num_classes=datamodule.num_classes)

    idx_to_class = {k: v for v, k in datamodule.data_train.class_to_idx.items()}
    model.idx_to_class = idx_to_class

    sm_training_env = get_training_env()

    print(":: Training ...")
    trainer = train(model, datamodule, sm_training_env)

    print(":: Saving Model Ckpt")
    save_last_ckpt(trainer, sm_model_dir)

    print(":: Saving Scripted Model")
    save_scripted_model(model, sm_model_dir)
