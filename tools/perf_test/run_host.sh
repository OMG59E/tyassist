cd ./output

cp $DENGINE_ROOT/tyhcp/platform/$HOST_PLAT/bin/iss_worker .
cp $DENGINE_ROOT/tyhcp/platform/$HOST_PLAT/bin/iss_config.txt .
cp $DENGINE_ROOT/tyhcp/platform/$HOST_PLAT/bin/main.bin .
cp $DENGINE_ROOT/tyhcp/platform/$HOST_PLAT/bin/nnp_main_cpp.lst .

export DCL_CFG_PATH=$DENGINE_ROOT/tyhcp/config/sdk.cfg
export DCL_DATA_PATH=$DENGINE_ROOT/tyexamples/data/datasets/ILSVRC2012/ILSVRC2012_img_val/ILSVRC2012_val_00000001.JPEG
export DCL_MODEL_PATH=$DENGINE_ROOT/tyexamples/models/dp2000/caffe_squeezenet_v1.1/net_combine.bin

if [ $1 ]; then
  TEST_NUM=$1
else
  TEST_NUM=1
fi

if [ $2 ]; then
  THREAD_NUM=$2
else
  THREAD_NUM=1
fi

mkdir -p ./result

# 参数可通过环境变量或运行参数指定
./perf_test -c $DCL_CFG_PATH -d $DCL_DATA_PATH -m $DCL_MODEL_PATH -r $TEST_NUM -t $THREAD_NUM