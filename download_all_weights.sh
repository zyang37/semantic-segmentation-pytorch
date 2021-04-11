echo "Downloading pretrained weights..."
MODEL_PATH=ckpt/

MODEL_NAME1=ade20k-resnet101dilated-ppm_deepsup
ENCODER1=$MODEL_NAME1/encoder_epoch_25.pth
DECODER1=$MODEL_NAME1/decoder_epoch_25.pth

MODEL_NAME2=ade20k-hrnetv2-c1
ENCODER2=$MODEL_NAME2/encoder_epoch_30.pth
DECODER2=$MODEL_NAME2/decoder_epoch_30.pth

MODEL_NAME3=ade20k-mobilenetv2dilated-c1_deepsup
ENCODER3=$MODEL_NAME3/encoder_epoch_20.pth
DECODER3=$MODEL_NAME3/decoder_epoch_20.pth

#MODEL_NAME4=ade20k-resnet18dilated-c1_deepsup
#ENCODER4=$MODEL_NAME4/encoder_epoch_20.pth
#DECODER4=$MODEL_NAME4/decoder_epoch_20.pth

MODEL_NAME5=ade20k-resnet18dilated-ppm_deepsup
ENCODER5=$MODEL_NAME5/encoder_epoch_20.pth
DECODER5=$MODEL_NAME5/decoder_epoch_20.pth

MODEL_NAME6=ade20k-resnet50-upernet
ENCODER6=$MODEL_NAME6/encoder_epoch_30.pth
DECODER6=$MODEL_NAME6/decoder_epoch_30.pth

MODEL_NAME7=ade20k-resnet50dilated-ppm_deepsup
ENCODER7=$MODEL_NAME7/encoder_epoch_20.pth
DECODER7=$MODEL_NAME7/decoder_epoch_20.pth

MODEL_NAME8=ade20k-resnet101-upernet
ENCODER8=$MODEL_NAME8/encoder_epoch_50.pth
DECODER8=$MODEL_NAME8/decoder_epoch_50.pth

if [ ! -e $MODEL_PATH ]; then
  mkdir -p $MODEL_PATH
fi

# model 1
MODEL_PATH=ckpt/$MODEL_NAME1
mkdir -p $MODEL_PATH
if [ ! -e $ENCODER1 ]; then
  wget -P $MODEL_PATH http://sceneparsing.csail.mit.edu/model/pytorch/$ENCODER1
fi
if [ ! -e $DECODER1 ]; then
  wget -P $MODEL_PATH http://sceneparsing.csail.mit.edu/model/pytorch/$DECODER1
fi

# model 2
MODEL_PATH=ckpt/$MODEL_NAME2
mkdir -p $MODEL_PATH
if [ ! -e $ENCODER2 ]; then
  wget -P $MODEL_PATH http://sceneparsing.csail.mit.edu/model/pytorch/$ENCODER2
fi
if [ ! -e $DECODER2 ]; then
  wget -P $MODEL_PATH http://sceneparsing.csail.mit.edu/model/pytorch/$DECODER2
fi

# model 3
MODEL_PATH=ckpt/$MODEL_NAME3
mkdir -p $MODEL_PATH
if [ ! -e $ENCODER3 ]; then
  wget -P $MODEL_PATH http://sceneparsing.csail.mit.edu/model/pytorch/$ENCODER3
fi
if [ ! -e $DECODER3 ]; then
  wget -P $MODEL_PATH http://sceneparsing.csail.mit.edu/model/pytorch/$DECODER3
fi

# model 5
MODEL_PATH=ckpt/$MODEL_NAME5
mkdir -p $MODEL_PATH
if [ ! -e $ENCODER5 ]; then
  wget -P $MODEL_PATH http://sceneparsing.csail.mit.edu/model/pytorch/$ENCODER5
fi
if [ ! -e $DECODER5 ]; then
  wget -P $MODEL_PATH http://sceneparsing.csail.mit.edu/model/pytorch/$DECODER5
fi

# model 6
MODEL_PATH=ckpt/$MODEL_NAME6
mkdir -p $MODEL_PATH
if [ ! -e $ENCODER6 ]; then
  wget -P $MODEL_PATH http://sceneparsing.csail.mit.edu/model/pytorch/$ENCODER6
fi
if [ ! -e $DECODER6 ]; then
  wget -P $MODEL_PATH http://sceneparsing.csail.mit.edu/model/pytorch/$DECODER6
fi

# model 7
MODEL_PATH=ckpt/$MODEL_NAME7
mkdir -p $MODEL_PATH
if [ ! -e $ENCODER7 ]; then
  wget -P $MODEL_PATH http://sceneparsing.csail.mit.edu/model/pytorch/$ENCODER7
fi
if [ ! -e $DECODER7 ]; then
  wget -P $MODEL_PATH http://sceneparsing.csail.mit.edu/model/pytorch/$DECODER7
fi

# model 8
MODEL_PATH=ckpt/$MODEL_NAME8
mkdir -p $MODEL_PATH
if [ ! -e $ENCODER8 ]; then
  wget -P $MODEL_PATH http://sceneparsing.csail.mit.edu/model/pytorch/$ENCODER8
fi
if [ ! -e $DECODER8 ]; then
  wget -P $MODEL_PATH http://sceneparsing.csail.mit.edu/model/pytorch/$DECODER8
fi