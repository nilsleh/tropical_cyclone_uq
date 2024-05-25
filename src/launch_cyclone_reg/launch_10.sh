#!/bin/bash
CUDA_VISIBLE_DEVICES=3
python run_cli_script.py model_config=configs/cyclone_reg/swag.yaml data_config=configs/cyclone_reg/dataset.yaml trainer_config=configs/cyclone_reg/trainer.yaml experiment.seed=10 trainer.devices=[0]