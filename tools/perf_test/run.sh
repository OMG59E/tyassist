export DCL_CFG_PATH=$DENGINE_ROOT/tyhcp/config/sdk.cfg
export DCL_MODEL_PATH=$DENGINE_ROOT/tyexamples/models/dp2000/caffe_squeezenet_v1.1/net_combine.bin
export DCL_DATA_PATH=$DENGINE_ROOT/tyexamples/data/bin/COCO_val2014_000000000139.jpg.416x416.rgb.plane.bin

cd ./output
mkdir -p ./result

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

# 参数可通过环境变量或运行参数指定
./perf_test -c $DCL_CFG_PATH -d $DCL_DATA_PATH -m $DCL_MODEL_PATH -r $TEST_NUM -t $THREAD_NUM
