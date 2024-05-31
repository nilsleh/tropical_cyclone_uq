#!/bin/bash
CUDA_VISIBLE_DEVICES=0
python run_cli_script_pv.py model_config=configs/pv_reg/due2.yaml data_config=configs/pv_reg/dataset.yaml trainer_config=configs/pv_reg/trainer.yaml experiment.seed=5 trainer.devices=[0]