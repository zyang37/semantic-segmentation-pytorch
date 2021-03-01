#!/bin/bash
#python3 noise_eval_one.py --cfg config/ade20k-hrnetv2.yaml -c 200 -s tmp_results/sp200_hrnetv2.csv
#python3 noise_eval_one.py --cfg config/ade20k-mobilenetv2dilated-c1_deepsup.yaml -c 200 -s tmp_results/sp200_mobilenetv2dilated-c1_deepsup.csv
#python3 noise_eval_one.py --cfg config/ade20k-resnet101dilated-ppm_deepsup.yaml -c 200 -s tmp_results/sp200_resnet101dilated-ppm_deepsup.csv
#python3 noise_eval_one.py --cfg config/ade20k-resnet101-upernet.yaml -c 200 -s tmp_results/sp200_resnet101-upernet.csv
#python3 noise_eval_one.py --cfg config/ade20k-resnet18dilated-ppm_deepsup.yaml -c 200 -s tmp_results/sp200_resnet18dilated-ppm_deepsup.csv
#python3 noise_eval_one.py --cfg config/ade20k-resnet50dilated-ppm_deepsup.yaml -c 200 -s tmp_results/sp200_resnet50dilated-ppm_deepsup.csv
#python3 noise_eval_one.py --cfg config/ade20k-resnet50-upernet.yaml -c 200 -s tmp_results/sp200_resnet50-upernet.csv

python3 noise_eval_one.py --cfg config/ade20k-hrnetv2.yaml -c 150 -s tmp_results/sp150_hrnetv2.csv
python3 noise_eval_one.py --cfg config/ade20k-mobilenetv2dilated-c1_deepsup.yaml -c 150 -s tmp_results/sp150_mobilenetv2dilated-c1_deepsup.csv
python3 noise_eval_one.py --cfg config/ade20k-resnet101dilated-ppm_deepsup.yaml -c 150 -s tmp_results/sp150_resnet101dilated-ppm_deepsup.csv
python3 noise_eval_one.py --cfg config/ade20k-resnet101-upernet.yaml -c 150 -s tmp_results/sp150_resnet101-upernet.csv
python3 noise_eval_one.py --cfg config/ade20k-resnet18dilated-ppm_deepsup.yaml -c 150 -s tmp_results/sp150_resnet18dilated-ppm_deepsup.csv
python3 noise_eval_one.py --cfg config/ade20k-resnet50dilated-ppm_deepsup.yaml -c 150 -s tmp_results/sp150_resnet50dilated-ppm_deepsup.csv
python3 noise_eval_one.py --cfg config/ade20k-resnet50-upernet.yaml -c 150 -s tmp_results/sp150_resnet50-upernet.csv