########################################### GPU ###########################################
mkdir -p tmp_results/

echo "(time) GPU..."
echo "(time) using GPU..." > inference_time.res
echo "(time) hrnetv2" >> inference_time.res
python3 predict_img.py -d 0 -i teaser/car_detection_sample.png --cfg  config/ade20k-hrnetv2.yaml >> inference_time.res
#sleep 2

echo "(time) mobilenetv2dilated" >> inference_time.res
python3 predict_img.py -d 0 -i teaser/car_detection_sample.png --cfg  config/ade20k-mobilenetv2dilated-c1_deepsup.yaml >> inference_time.res
#sleep 2

echo "(time) resnet101dilated" >> inference_time.res
python3 predict_img.py -d 0 -i teaser/car_detection_sample.png --cfg  config/ade20k-resnet101dilated-ppm_deepsup.yaml >> inference_time.res
#sleep 2

echo "(time) resnet101" >> inference_time.res
python3 predict_img.py -d 0 -i teaser/car_detection_sample.png --cfg  config/ade20k-resnet101-upernet.yaml >> inference_time.res
#sleep 2

echo "(time) resnet18dilated" >> inference_time.res
python3 predict_img.py -d 0 -i teaser/car_detection_sample.png --cfg  config/ade20k-resnet18dilated-ppm_deepsup.yaml >> inference_time.res
#sleep 2

echo "(time) resnet50dilated-ppm" >> inference_time.res
python3 predict_img.py -d 0 -i teaser/car_detection_sample.png --cfg  config/ade20k-resnet50dilated-ppm_deepsup.yaml >> inference_time.res
#sleep 2

echo "(time) resnet50dilated-upernet" >> inference_time.res
python3 predict_img.py -d 0 -i teaser/car_detection_sample.png --cfg  config/ade20k-resnet50-upernet.yaml >> inference_time.res
#sleep 2


########################################### CPU ###########################################
echo "(time) CPU..."
echo "(time) using CPU..." >> inference_time.res
echo "(time) hrnetv2" >> inference_time.res
CUDA_VISIBLE_DEVICES="" python3 predict_img.py -d 0 -i teaser/car_detection_sample.png --cfg  config/ade20k-hrnetv2.yaml >> inference_time.res

echo "(time) mobilenetv2dilated" >> inference_time.res
CUDA_VISIBLE_DEVICES="" python3 predict_img.py -d 0 -i teaser/car_detection_sample.png --cfg  config/ade20k-mobilenetv2dilated-c1_deepsup.yaml >> inference_time.res

echo "(time) resnet101dilated" >> inference_time.res
CUDA_VISIBLE_DEVICES="" python3 predict_img.py -d 0 -i teaser/car_detection_sample.png --cfg  config/ade20k-resnet101dilated-ppm_deepsup.yaml >> inference_time.res

echo "(time) resnet101" >> inference_time.res
CUDA_VISIBLE_DEVICES="" python3 predict_img.py -d 0 -i teaser/car_detection_sample.png --cfg  config/ade20k-resnet101-upernet.yaml >> inference_time.res

echo "(time) resnet18dilated" >> inference_time.res
CUDA_VISIBLE_DEVICES="" python3 predict_img.py -d 0 -i teaser/car_detection_sample.png --cfg  config/ade20k-resnet18dilated-ppm_deepsup.yaml >> inference_time.res

echo "(time) resnet50dilated-ppm" >> inference_time.res
CUDA_VISIBLE_DEVICES="" python3 predict_img.py -d 0 -i teaser/car_detection_sample.png --cfg  config/ade20k-resnet50dilated-ppm_deepsup.yaml >> inference_time.res

echo "(time) resnet50dilated-upernet" >> inference_time.res
CUDA_VISIBLE_DEVICES="" python3 predict_img.py -d 0 -i teaser/car_detection_sample.png --cfg  config/ade20k-resnet50-upernet.yaml >> inference_time.res