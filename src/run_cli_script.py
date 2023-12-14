import os
from datetime import datetime
from typing import Any
import torch

from hydra.utils import instantiate
from hydra.errors import InstantiationException
from lightning import Trainer
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from omegaconf.errors import ConfigAttributeError
from lightning.pytorch.loggers import CSVLogger, WandbLogger  # noqa: F401
from omegaconf import OmegaConf


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
        WandbLogger(
            name=config["experiment"]["experiment_name"],
            save_dir=config["experiment"]["save_dir"],
            project=config["wandb"]["project"],
            entity=config["wandb"]["entity"],
            resume="allow",
            mode="offline",
        ),
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

    lr_monitor_callback = LearningRateMonitor(logging_interval="step")

    return instantiate(
        config.trainer,
        default_root_dir=config["experiment"]["save_dir"],
        callbacks=[checkpoint_callback, lr_monitor_callback],
        # callbacks=[lr_monitor_callback],
        logger=loggers,
    )

post_hoc_methods = ["SWAG", "Laplace", "ConformalQR", "CARD", "DeepEnsemble"]


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    command_line_conf = OmegaConf.from_cli()
    model_conf = OmegaConf.load(command_line_conf.model_config)
    data_conf = OmegaConf.load(command_line_conf.data_config)
    trainer_conf = OmegaConf.load(command_line_conf.trainer_config)

    full_config = OmegaConf.merge(data_conf, trainer_conf, model_conf)
    full_config = create_experiment_dir(full_config)

    datamodule = instantiate(full_config.datamodule)

    trainer = generate_trainer(full_config)
    
    if any(method in full_config.uq_method._target_ for method in post_hoc_methods):
        # post hoc methods just load a checkpoint
        if "SWAG" in full_config.uq_method["_target_"] or "CARD" in full_config.uq_method["_target_"]:
            model = instantiate(full_config.uq_method)
            trainer.fit(model, datamodule=datamodule)
        elif "Laplace" in full_config.uq_method["_target_"]:
            pass
        elif "DeepEnsemble" in full_config.uq_method["_target_"]:
            ensemble_members = [
                {"base_model": instantiate(full_config.ensemble_members), "ckpt_path": path}
                for path in full_config.uq_method.ensemble_members
            ]
            model = instantiate(full_config.uq_method, ensemble_members=ensemble_members)
        elif "ConformalQR" in full_config.uq_method["_target_"]:
            datamodule.setup("fit")
            model = instantiate(full_config.uq_method)
            trainer.validate(model, dataloaders=datamodule.calibration_dataloader())
        else:
            model = instantiate(full_config.uq_method)
            trainer.validate(model, datamodule=datamodule)

        trainer.test(model, datamodule=datamodule)
    else:
        try:
            model = instantiate(full_config.uq_method.model, pretrained=False, num_classes=1000)
        except ConfigAttributeError:
            model = instantiate(full_config.uq_method.feature_extractor, pretrained=False, num_classes=1000)
            num_classes = full_config.uq_method.feature_extractor.num_classes
        
        prev_conv1 = model.conv1.weight.data.clone()
        model.load_state_dict(torch.load(full_config.resnet_ckpt))
        print(f"Weights are loaded if the first layer is not equal anymore, so torch equal should be False, got: {torch.equal(prev_conv1, model.conv1.weight.data)}")
        # replace last layer
        model.fc = torch.nn.Linear(
            in_features=model.fc.in_features,
            out_features=num_classes,
            bias=True,
        )

        try:
            model = instantiate(full_config.uq_method, model=model)
        except:
            model = instantiate(full_config.uq_method, feature_extractor=model)

        trainer.fit(model, datamodule)
        trainer.test(ckpt_path="best", datamodule=datamodule)

    # store predictions for training and test set
    target_mean = datamodule.target_mean.cpu()
    target_std = datamodule.target_std.cpu()

    # Also store predictions for training
    def collate(batch: list[dict[str, torch.Tensor]]):
        """Collate fn to include augmentations."""
        images = [item["input"] for item in batch]
        labels = [item["target"] for item in batch]

        inputs = torch.stack(images)
        targets = torch.stack(labels)
        if datamodule.task == "regression":
            return {
                "input": datamodule.aug({"image": inputs.float()})["image"],
                "target": (targets[..., -1:].float() - target_mean) / target_std,
            }
        else:
            return {
                "input": datamodule.aug({"image": inputs.float()})["image"],
                "target": targets.squeeze().long(),
            }

    # train dataset results
    model.pred_file_name = "preds_train.csv"
    datamodule.setup("fit")
    train_loader = datamodule.train_dataloader()
    train_loader.shuffle = False
    train_loader.collate_fn = collate

    try:
        trainer.test(ckpt_path="best", dataloaders=train_loader)
    except:
        trainer.test(model, dataloaders=train_loader)

    # val dataset results
    model.pred_file_name = "preds_val.csv"
    val_loader = datamodule.val_dataloader()
    val_loader.shuffle = False
    val_loader.collate_fn = collate

    try:
        trainer.test(ckpt_path="best", dataloaders=val_loader)
    except:
        trainer.test(model, dataloaders=val_loader)

    # save configuration file
    with open(
        os.path.join(full_config["experiment"]["save_dir"], "config.yaml"), "w"
    ) as f:
        OmegaConf.save(config=full_config, f=f)

    print("FINISHED EXPERIMENT", flush=True)
