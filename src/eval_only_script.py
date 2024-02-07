from omegaconf import OmegaConf
import torch
import glob
import os
from hydra.utils import instantiate
from torch.utils.data import DataLoader


def make_predictions(dataset_type, model, datamodule, trainer, full_config):
    assert dataset_type in [
        "train",
        "val",
        "test",
    ], "Invalid dataset type. Choose from 'train', 'val', 'test'."

    target_mean = datamodule.target_mean.cpu()
    target_std = datamodule.target_std.cpu()

    # Define the collate function
    def collate(batch: list[dict[str, torch.Tensor]]):
        """Collate fn to include augmentations."""
        images = [item["input"] for item in batch]
        labels = [item["target"] for item in batch]

        inputs = torch.stack(images)
        targets = torch.stack(labels)
        if datamodule.task == "regression":
            new_batch = {
                "input": datamodule.aug({"input": inputs.float()})["input"],
                "target": (targets.float() - target_mean) / target_std,
            }
        else:
            new_batch = {
                "input": datamodule.aug({"input": inputs.float()})["input"],
                "target": targets.long(),
            }

        # add back all other keys
        for item in batch:
            for key, value in item.items():
                if key not in ["input", "target"]:
                    new_batch.setdefault(key, []).append(value)
        return new_batch

    # Set the prediction file name based on the dataset type
    model.pred_file_name = f"preds_{dataset_type}.csv"

    # Get the appropriate data loader based on the dataset type
    if dataset_type == "train":
        datamodule.setup("fit")
        data_loader = datamodule.train_dataloader()
    elif dataset_type == "val":
        data_loader = datamodule.val_dataloader()
    else:  # 'test'
        datamodule.setup("test")
        data_loader = datamodule.test_dataloader()

    # print(f"LENGTH {dataset_type}", len(data_loader.dataset))
    # Set the shuffle and collate_fn attributes of the data loader
    # Create a new DataLoader with shuffle=False
    data_loader = DataLoader(
        data_loader.dataset,
        batch_size=data_loader.batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=data_loader.num_workers,
    )

    # Test the model
    trainer.test(model, dataloaders=data_loader)


# TODO DKL, DUE, Deep Ensembel

if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")

    # based on an experimnet directory find configs and checkpoints
    exp_dir = "/p/project/hai_uqmethodbox/experiment_output/min_card"
    new_exp_dir = "/p/project/hai_uqmethodbox/experiment_output/min_wind_speed_0_debug"

    # make a dictionary of config and checkpoint paths
    config_paths = glob.glob(os.path.join(exp_dir, "*", "config.yaml"))

    # find checkpoints contained in same directory as config
    config_ckpt_dict = {}
    for config_path in config_paths:
        name = os.path.basename(os.path.dirname(config_path)).split("_")[1]
        config_ckpt_dict[name] = {}
        config_ckpt_dict[name]["config_path"] = config_path
        try:
            config_ckpt_dict[name]["ckpt_path"] = glob.glob(
                os.path.join(os.path.dirname(config_path), "*.ckpt")
            )[0]
        except:
            config_ckpt_dict[name]["ckpt_path"] = None

    # loop over configs and checkpoints
    for name, paths in config_ckpt_dict.items():

        # load config
        config = OmegaConf.load(paths["config_path"])

        # in the config need to update experiment save dir and exp_dir
        config.experiment.exp_dir = new_exp_dir
        config.experiment.save_dir = os.path.join(
            new_exp_dir, os.path.basename(os.path.dirname(paths["config_path"]))
        )

        # if os.path.exists(config.experiment.save_dir):
        #     continue

        if "DeepEnsemble" in config.uq_method["_target_"]:
            ensemble_members = [
                {"base_model": instantiate(config.ensemble_members), "ckpt_path": path}
                for path in config.uq_method.ensemble_members
            ]
            model = instantiate(config.uq_method, ensemble_members=ensemble_members)
        else:
            model = instantiate(config.uq_method)

        if paths["ckpt_path"] is not None:
            ckpt = torch.load(paths["ckpt_path"], map_location="cpu")

            try:
                match_message = model.load_state_dict(ckpt["state_dict"])
            except RuntimeError:
                print("FAIL")
                continue
            if match_message.missing_keys or match_message.unexpected_keys:
                import pdb

                pdb.set_trace()
                print("Missing keys:", match_message.missing_keys)
                print("Unexpected keys:", match_message.unexpected_keys)
                continue

        # load data
        datamodule = instantiate(config.datamodule)

        target_mean = datamodule.target_mean.cpu()
        target_std = datamodule.target_std.cpu()

        def collate(batch: list[dict[str, torch.Tensor]]):
            """Collate fn to include augmentations."""
            images = [item["input"] for item in batch]
            labels = [item["target"] for item in batch]

            inputs = torch.stack(images)
            targets = torch.stack(labels)
            if datamodule.task == "regression":
                new_batch = {
                    "input": datamodule.aug({"input": inputs.float()})["input"],
                    "target": (targets.float() - target_mean) / target_std,
                }
            else:
                new_batch = {
                    "input": datamodule.aug({"input": inputs.float()})["input"],
                    "target": targets.long(),
                }

            # add back all other keys
            for item in batch:
                for key, value in item.items():
                    if key not in ["input", "target"]:
                        new_batch.setdefault(key, []).append(value)
            return new_batch

        os.makedirs(config.experiment.save_dir, exist_ok=True)

        # load trainer
        config.trainer.accelerator = "gpu"
        config.trainer.devices = [0]
        config.trainer.default_root_dir = config.experiment.save_dir
        trainer = instantiate(config.trainer)

        if (
            "SWAG" in config.uq_method["_target_"]
            or "Laplace" in config.uq_method["_target_"]
        ):
            trainer.fit(model, datamodule=datamodule)
        elif "ConformalQR" in config.uq_method["_target_"]:
            datamodule.setup("fit")
            calib_loader = datamodule.calibration_dataloader()
            calib_loader.collate_fn = collate
            model = instantiate(config.uq_method)
            trainer.validate(model, dataloaders=calib_loader)

        # make predictions
        make_predictions("test", model, datamodule, trainer, config)
        make_predictions("val", model, datamodule, trainer, config)
        make_predictions("train", model, datamodule, trainer, config)

        # copy config to save dir with omegaconf
        OmegaConf.save(config, os.path.join(config.experiment.save_dir, "config.yaml"))
