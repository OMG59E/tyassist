/**
* @file main.cpp
*
* Copyright (C) 2020. intellif Technologies Co., Ltd. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/
#include <iostream>
#include "device_process.h"
#include "utils.h"
#include "model_process.h"
#include "dcl_mdl.h"
#include <map>
#include "data_process.h"
#include <thread>
#include <unistd.h>
#include "get_time.h"
#include "random.h"

using namespace std;

int PostProcess(void *data[], int len[], int num) {
    INFO_LOG("empty post process, out num=%d", num);
}

typedef struct{
    int type = 0; // 0 - no interval  1 - fixed interval  2 - possion interval
    int interval = 0; // ms
}InterValConfig;


int benchmark_perf(char *dclCfgDir, char *dclDataDir, char *dclModelDir, int aippFlag, int totalNum, int thdNum, InterValConfig config, bool postprocess)
{
    std::string dclConfigPath = dclCfgDir;
    std::string dclDataDirPath = dclDataDir;
    std::string dclModelDirPath = dclModelDir;
    // 输入为bin文件时需要设定对应图像的长宽，输入为图像文件时会自动从文件中读取
    int width = 416;
    int height = 416;

    auto t1 = GetTime::Now();

    // init device
    deviceProcess deviceProc;
    deviceProc.InitResource(dclConfigPath);

    // load model   
    ModelProcess modelProcess{};
    Result ret = modelProcess.LoadModel(dclModelDirPath, aippFlag);
    if(ret != SUCCESS) {
        ERROR_LOG("load model fail, %s", dclModelDirPath.c_str());
        return -1;
    }
    INFO_LOG("load model success, %s", dclModelDirPath.c_str());

    // init model
    ret = modelProcess.init();
    if(ret != SUCCESS) {
        ERROR_LOG("init model fail, %s", dclModelDirPath.c_str());
        return -1;
    }
    INFO_LOG("init model success, %s", dclModelDirPath.c_str());

    // 查看模型输入输出信息
    modelProcess.PrintIOInfo(aippFlag);

    // 初始化数据
    vector<string> testFile;
    testFile.push_back(dclDataDirPath);
    DataProcess dataProc;
    if (aippFlag) {
        AippInfo aippInfo;
        aippInfo.width = width;
        aippInfo.height = height;
        aippInfo.format = DCL_PIXEL_FORMAT_RGB_888_PLANAR;
        dataProc.Init(testFile, modelProcess.modelDesc_, modelProcess.modelId_, &aippInfo);
    } else {
        dataProc.Init(testFile, modelProcess.modelDesc_, modelProcess.modelId_);
    }
    dclmdlDataset *input = dataProc.GetInputDataSet();
    dclmdlDataset *output = dataProc.GetOutputDataSet();

    std::vector<std::thread> threads;
    std::vector<double> times[thdNum];

    
    for(int j = 0; j < thdNum; j++) 
    { 
        int testNum = (j==thdNum-1)?(totalNum/thdNum + totalNum%thdNum):(totalNum/thdNum);

        threads.push_back(
            std::thread([&](int thdId, int loopNum) {
                Random r(true);	
                
                for (int i = 0; i < loopNum; i++) {
                    auto t2 = GetTime::Now();
                    ret = modelProcess.ExecuteSync(modelProcess.modelId_, input, output);

                    if (postprocess) {
                        int outputNum = dclmdlGetDatasetNumBuffers(output);
                        void *data[outputNum];
                        int len[outputNum];
                        
                        for (size_t k = 0; k < outputNum; ++k) {
                            // get model output data
                            dclDataBuffer *dataBuffer = dclmdlGetDatasetBuffer(output, k);
                            
                            data[k] = dclGetDataBufferAddr(dataBuffer);
                            len[k] = dclGetDataBufferSize(dataBuffer);
                            // INFO_LOG("ExecuteSync output[%d], len=%d", k, len[k]);

                            // outData = reinterpret_cast<float *>(data);
                            // Utils::SaveBinfile(std::string(picName + ".out"), (char*)data, len);
                        }

                        // add postprocess here
                        PostProcess(data, len, outputNum);
                    }
                    auto t3 = GetTime::Now();
                    double timeMs = GetTime::DurationMs(t2, t3);
                    INFO_LOG("perf thread %d, test %d, cost %fms", thdId, i, timeMs);
                    times[thdId].push_back(timeMs);

                    if(config.type == 1)
                        std::this_thread::sleep_for(std::chrono::milliseconds(config.interval));
                    if(config.type == 2)
                        std::this_thread::sleep_for(std::chrono::milliseconds(r.poisson(config.interval)));
                }
                INFO_LOG("thread %d exit!", thdId);
            }, j, testNum)
        ); 
    } 

    for (int i = 0; i < thdNum; i++) 
    { 
        threads[i].join(); 
    }

    double totalAvg = 0.0;
    // 计算各线程耗时平均delay值
    for (int i = 0; i < thdNum; i++) {
        double thdAvg = 0.0;
        int num = times[i].size();
        for (int j = 0; j < num; j++) {
            thdAvg += times[i][j];
        }
        totalAvg += thdAvg;
        thdAvg /= num;
        INFO_LOG("perf thread %d test=%d, average cost %fms", i, num, thdAvg);
    }
    totalAvg /= totalNum;
    INFO_LOG("perf total test=%d, average cost %fms", totalNum, totalAvg);

    auto t4 = GetTime::Now();
    double_t total_cost_ms = GetTime::DurationMs(t1, t4);
    INFO_LOG("perf test success, all cost %fms, fps %d", total_cost_ms, totalNum*1000/total_cost_ms);

    return SUCCESS;
}


int main(int argc, char *argv[])
{
    char *dclCfgDirPath = getenv("DCL_CFG_PATH");
    char *dclDataDirPath = getenv("DCL_DATA_PATH");
    char *dclModelDirPath = getenv("DCL_MODEL_PATH");

    int totalNum = 1;
    int thdNum = 1;
    int aippFlag = 1;
    bool postprocess = false;
    InterValConfig config;
    int result;

    while ((result = getopt(argc, argv, "c:d:m:n:r:t:i:j:")) != -1)
    {
        switch (result)
        {
            case 'h':
                INFO_LOG("./perf_test");
                INFO_LOG("\t<-c /DEngine/tyhcp/config/sdk.cfg>");
                INFO_LOG("\t<-d /DEngine/tyexamples/data/datasets/ILSVRC2012/ILSVRC2012_img_val/ILSVRC2012_val_00000001.JPEG>");
                INFO_LOG("\t<-m /DEngine/tyexamples/models/dp2000/caffe_squeezenet_v1.1/net_combine.bin>");
                INFO_LOG("\t<-n> - disable aipp, default is enable.");
                INFO_LOG("\t<-p> - enable postprocess, default is disable.");
                INFO_LOG("\t<-r 1> - set test num, default is 1.");
                INFO_LOG("\t<-t 1> - set thread num, default is 1.");
                INFO_LOG("\t<-i 0> - set interval type, default is 0.");
                INFO_LOG("\t<-j 0> - set interval parameter, default is 0.");
                return 0;
            case 'c':
                dclCfgDirPath = optarg;
                break;
            case 'd':
                dclDataDirPath = optarg;
                break;
            case 'm':
                dclModelDirPath = optarg;
                break;
            case 'n':
                aippFlag = 0;
                break;
            case 'r':
                totalNum = atoi(optarg);
                break;
            case 't':
                thdNum = atoi(optarg);
                break;
            case 'p':
                postprocess = true;
                break;
            case 'i':
                config.type = atoi(optarg);
                break;
            case 'j':
                config.interval = atoi(optarg);
                break;
            default:
                printf("default, result=%c\n",result);
                break;
        }
    }
    INFO_LOG("config: dclConfigPath=%s, dclDataDirPath=%s, dclModelDirPath=%s, aippFlag=%d, loop=%d, thdnum=%d, intervalType=%d, intervalParam=%d postprocess=%d\n", 
        dclCfgDirPath, dclDataDirPath, dclModelDirPath, aippFlag, totalNum, thdNum, config.type, config.interval, postprocess);

    if ((nullptr == dclCfgDirPath) || (nullptr == dclDataDirPath) || (nullptr == dclModelDirPath)) {
        ERROR_LOG("please set environment variable, or add parameters, use -h for more details.");
        return -1;
    }

    benchmark_perf(dclCfgDirPath, dclDataDirPath, dclModelDirPath, aippFlag, totalNum, thdNum, config, postprocess);
    return 0;
}