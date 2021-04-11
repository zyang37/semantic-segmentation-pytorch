echo "(time) using GPU..." > tmp_results/inference_time

echo "(time) hrnetv2" >> tmp_results/inference_time
python3 predict_img.py -d 0 -i ../../Downloads/car_detection_sample1.png --cfg  config/ade20k-hrnetv2.yaml >> tmp_results/inference_time
sleep 5

echo "(time) mobilenetv2dilated" >> tmp_results/inference_time
python3 predict_img.py -d 0 -i ../../Downloads/car_detection_sample1.png --cfg  config/ade20k-mobilenetv2dilated-c1_deepsup.yaml >> tmp_results/inference_time
sleep 5

echo "(time) resnet101dilated" >> tmp_results/inference_time
python3 predict_img.py -d 0 -i ../../Downloads/car_detection_sample1.png --cfg  config/ade20k-resnet101dilated-ppm_deepsup.yaml >> tmp_results/inference_time
sleep 5

echo "(time) resnet101" >> tmp_results/inference_time
python3 predict_img.py -d 0 -i ../../Downloads/car_detection_sample1.png --cfg  config/ade20k-resnet101-upernet.yaml >> tmp_results/inference_time
sleep 5

echo "(time) resnet18dilated" >> tmp_results/inference_time
python3 predict_img.py -d 0 -i ../../Downloads/car_detection_sample1.png --cfg  config/ade20k-resnet18dilated-ppm_deepsup.yaml >> tmp_results/inference_time
sleep 5

echo "(time) resnet50dilated-ppm" >> tmp_results/inference_time
python3 predict_img.py -d 0 -i ../../Downloads/car_detection_sample1.png --cfg  config/ade20k-resnet50dilated-ppm_deepsup.yaml >> tmp_results/inference_time
sleep 5

echo "(time) resnet50dilated-upernet" >> tmp_results/inference_time
python3 predict_img.py -d 0 -i ../../Downloads/car_detection_sample1.png --cfg  config/ade20k-resnet50-upernet.yaml >> tmp_results/inference_time
sleep 5


echo "(time) using CPU..." >> tmp_results/inference_time
echo "(time) hrnetv2" >> tmp_results/inference_time
CUDA_VISIBLE_DEVICES="" python3 predict_img.py -d 0 -i ../../Downloads/car_detection_sample1.png --cfg  config/ade20k-hrnetv2.yaml >> tmp_results/inference_time

echo "(time) mobilenetv2dilated" >> tmp_results/inference_time
CUDA_VISIBLE_DEVICES="" python3 predict_img.py -d 0 -i ../../Downloads/car_detection_sample1.png --cfg  config/ade20k-mobilenetv2dilated-c1_deepsup.yaml >> tmp_results/inference_time

echo "(time) resnet101dilated" >> tmp_results/inference_time
CUDA_VISIBLE_DEVICES="" python3 predict_img.py -d 0 -i ../../Downloads/car_detection_sample1.png --cfg  config/ade20k-resnet101dilated-ppm_deepsup.yaml >> tmp_results/inference_time

echo "(time) resnet101" >> tmp_results/inference_time
CUDA_VISIBLE_DEVICES="" python3 predict_img.py -d 0 -i ../../Downloads/car_detection_sample1.png --cfg  config/ade20k-resnet101-upernet.yaml >> tmp_results/inference_time

echo "(time) resnet18dilated" >> tmp_results/inference_time
CUDA_VISIBLE_DEVICES="" python3 predict_img.py -d 0 -i ../../Downloads/car_detection_sample1.png --cfg  config/ade20k-resnet18dilated-ppm_deepsup.yaml >> tmp_results/inference_time

echo "(time) resnet50dilated-ppm" >> tmp_results/inference_time
CUDA_VISIBLE_DEVICES="" python3 predict_img.py -d 0 -i ../../Downloads/car_detection_sample1.png --cfg  config/ade20k-resnet50dilated-ppm_deepsup.yaml >> tmp_results/inference_time

echo "(time) resnet50dilated-upernet" >> tmp_results/inference_time
CUDA_VISIBLE_DEVICES="" python3 predict_img.py -d 0 -i ../../Downloads/car_detection_sample1.png --cfg  config/ade20k-resnet50-upernet.yaml >> tmp_results/inference_time