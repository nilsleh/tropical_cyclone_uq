#!/bin/bash
CUDA_VISIBLE_DEVICES=0
python run_cli_script.py model_config=configs/cyclone_reg/mc_dropout.yaml data_config=configs/cyclone_reg/dataset.yaml trainer_config=configs/cyclone_reg/trainer.yaml experiment.seed=10 trainer.devices=[0]