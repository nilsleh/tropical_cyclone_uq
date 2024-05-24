#!/bin/bash
CUDA_VISIBLE_DEVICES=1
python run_cli_script_pv.py model_config=configs/pv_reg/bnn_elbo.yaml data_config=configs/pv_reg/dataset.yaml trainer_config=configs/pv_reg/trainer.yaml experiment.seed=0 trainer.devices=[0]