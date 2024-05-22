#!/bin/bash
CUDA_VISIBLE_DEVICES=0
python run_cli_script.py model_config=configs/pv_reg/mc_dropout.yaml data_config=configs/pv_reg/dataset.yaml trainer_config=configs/pv_reg/trainer.yaml experiment.seed=0 trainer.devices=[0]