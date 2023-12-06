import os
from datetime import datetime
from typing import Any
import torch

from hydra.utils import instantiate
from lightning import Trainer
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import CSVLogger, WandbLogger  # noqa: F401
from omegaconf import OmegaConf
from tropical_cyclone_dm import TropicalCycloneSequenceDataModule


def create_experiment_dir(config: dict[str, Any]) -> str:
    """Create experiment directory.

    Args:
        config: config file

    Returns:
        config with updated save_dir
    """
    os.makedirs(config["experiment"]["exp_dir"], exist_ok=True)
    exp_dir_name = (
        f"{config['experiment']['experiment_name']}"
        f"_{config['uq_method']['_target_'].split('.')[-1]}"
        f"_{datetime.now().strftime('%m-%d-%Y_%H-%M-%S-%f')}"
    )
    config["experiment"]["experiment_name"] = exp_dir_name
    exp_dir_path = os.path.join(config["experiment"]["exp_dir"], exp_dir_name)
    os.makedirs(exp_dir_path)
    config["experiment"]["save_dir"] = exp_dir_path
    config["trainer"]["default_root_dir"] = exp_dir_path
    return config


def generate_trainer(config: dict[str, Any]) -> Trainer:
    """Generate a pytorch lightning trainer."""
    loggers = [
        CSVLogger(config["experiment"]["save_dir"], name="csv_logs"),
        # WandbLogger(
        #     name=config["experiment"]["experiment_name"],
        #     save_dir=config["experiment"]["save_dir"],
        #     project=config["wandb"]["project"],
        #     entity=config["wandb"]["entity"],
        #     resume="allow",
        #     config=config,
        #     mode=config["wandb"].get("mode", "online"),
        # ),
    ]

    track_metric = "val_loss"
    mode = "min"

    checkpoint_callback = ModelCheckpoint(
        dirpath=config["experiment"]["save_dir"],
        save_top_k=1,
        monitor=track_metric,
        mode=mode,
        every_n_epochs=1,
    )

    early_stopping_callback = EarlyStopping(
        monitor=track_metric, min_delta=1e-5, patience=100, mode=mode
    )

    lr_monitor_callback = LearningRateMonitor(logging_interval="step")

    return instantiate(
        config.trainer,
        default_root_dir=config["experiment"]["save_dir"],
        callbacks=[checkpoint_callback, early_stopping_callback, lr_monitor_callback],
        logger=loggers,
    )


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    command_line_conf = OmegaConf.from_cli()
    model_conf = OmegaConf.load(command_line_conf.model_config)
    data_conf = OmegaConf.load(command_line_conf.data_config)
    trainer_conf = OmegaConf.load(command_line_conf.trainer_config)

    full_config = OmegaConf.merge(data_conf, trainer_conf, model_conf)
    full_config = create_experiment_dir(full_config)

    datamodule = instantiate(full_config.datamodule)
    model = instantiate(full_config.uq_method)
    trainer = generate_trainer(full_config)

    # laplace only uses test
    if "laplace" not in command_line_conf.model_config:
        trainer.fit(model, datamodule)
        trainer.test(ckpt_path="best", datamodule=datamodule)
    else:
        trainer.test(model, datamodule=datamodule)

    target_mean = datamodule.target_mean.cpu()
    target_std = datamodule.target_std.cpu()

    # Also store predictions for training
    def train_collate(batch: list[dict[str, torch.Tensor]]):
        """Collate fn to include augmentations."""
        images = [item["input"] for item in batch]
        labels = [item["target"] for item in batch]

        inputs = torch.stack(images)
        targets = torch.stack(labels)
        return {
            "input": datamodule.aug({"image": inputs.float()})["image"],
            "target": (targets[..., -1:].float() - target_mean) / target_std,
        }

    model.pred_file_name = "predictions_train.csv"
    datamodule.setup("fit")
    train_loader = datamodule.train_dataloader()
    train_loader.shuffle = False
    train_loader.collate_fn = train_collate
    try:
        trainer.test(ckpt_path="best", dataloaders=train_loader)
    except:
        trainer.test(model, dataloaders=train_loader)

    with open(
        os.path.join(full_config["experiment"]["save_dir"], "config.yaml"), "w"
    ) as f:
        OmegaConf.save(config=full_config, f=f)
