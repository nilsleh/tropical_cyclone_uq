import itertools
import os
import stat
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="create_launch_scripts.py", description="Create bash scripts for slurm."
    )

    parser.add_argument(
        "--user", help="Name of user", type=str, required=True, choices=["nina", "nils"]
    )

    args = parser.parse_args()

    GPUS = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2]
    MODEL_CONF_FILE_NAMES = [
        "mc_dropout.yaml",
        "bnn_elbo.yaml",
        "der.yaml",
        "dkl.yaml",
        "due.yaml",
        "quantile_regression.yaml",
        "mve.yaml"
    ]
    CONF_BASE_DIR = f"/p/project/hai_uqmethodbox/{args.user}/tropical_cyclone_uq/src/configs/cyclone"
    SEEDS = [0]

    for idx, (seed, conf_name) in enumerate(itertools.product(SEEDS, MODEL_CONF_FILE_NAMES)):
        model_config_file = os.path.join(CONF_BASE_DIR, conf_name)
        data_config_file = os.path.join(CONF_BASE_DIR, "dataset.yaml")
        trainer_config_file = os.path.join(CONF_BASE_DIR, "trainer.yaml")
        command = (
            "python run_cli_script.py"
            + f" model_config={model_config_file}"
            + f" data_config={data_config_file}"
            + f" trainer_config={trainer_config_file}"
            + f" experiment.seed={seed}"
            + " trainer.devices=[0]"
        )

        base_dir = os.path.basename(CONF_BASE_DIR)

        command = command.strip()

        script = "#!/bin/bash\n"
        script += f"CUDA_VISIBLE_DEVICES={GPUS[idx]}\n"
        script += f"{command}"

        script_path = os.path.join(f"launch_{base_dir}", f"launch_{idx}.sh")
        with open(script_path, "w") as file:
            file.write(script)

        # make bash script exectuable
        st = os.stat(script_path)
        os.chmod(script_path, st.st_mode | stat.S_IEXEC)