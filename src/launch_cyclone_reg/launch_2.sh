#!/bin/bash
CUDA_VISIBLE_DEVICES=2
python run_cli_script.py model_config=configs/cyclone_reg/laplace.yaml data_config=configs/cyclone_reg/dataset.yaml trainer_config=configs/cyclone_reg/trainer.yaml experiment.seed=11 trainer.devices=[0]