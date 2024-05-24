#!/bin/bash
CUDA_VISIBLE_DEVICES=2
python run_cli_script_pv.py model_config=configs/pv_reg/laplace.yaml data_config=configs/pv_reg/dataset.yaml trainer_config=configs/pv_reg/trainer.yaml experiment.seed=11 trainer.devices=[0]