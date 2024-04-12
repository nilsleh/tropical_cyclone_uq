# tropical_cyclone_uq
Code for paper: "Uncertainty Aware Tropical Cyclone Wind Speed Estimation from Satellite Data" at ICLR Workshop Machine Learning for Remote Sensing 2024.

# Installation

```
pip install lightning-uq-box
pip install torchgeo
pip install wandb
```

# Run Experiments
The `run_cli_script.py` executes the experiment and expects a config file.

For example to run a BNN model.

```
python run_cli_script.py model_config=configs/cyclone_reg/bnn_elob.yaml data_config=configs/cyclone_reg/dataset.yaml trainer_config=configs/cyclone_reg/trainer.yaml experiment.seed=63 trainer.devices=[0]`
```

# Analysis
The plots in the paper are all generated with the `results_tropical.ipynb` notebook that expects a directory where the experiments are stored.


