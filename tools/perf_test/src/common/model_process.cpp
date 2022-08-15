/**
* @file model_process.cpp
*
* Copyright (C) 2020. intellif Technologies Co., Ltd. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/
#include "model_process.h"
#include <iostream>
#include <map>
#include <sstream>
#include <algorithm>
#include <functional>
#include "utils.h"
//#include "yolov2_post.h"


using namespace std;
extern bool g_isDevice;
extern size_t g_executeTimes;
extern size_t g_callbackInterval;

ModelProcess::ModelProcess() :modelId_(0), modelWorkSize_(0), modelWeightSize_(0),
    modelWorkPtr_(nullptr), modelWeightPtr_(nullptr), loadFlag_(false), modelDesc_(nullptr)
{
    //stream_ = stream;
}

ModelProcess::~ModelProcess()
{
    UnloadModel();
    DestroyModelDesc();
}

Result ModelProcess::LoadModel(const string &modelPath, int aippFlag)
{
    if (loadFlag_) {
        ERROR_LOG("model has already been loaded");
        return FAILED;
    }

    void *mdlData;
    uint32_t mdlSize;
    Utils::ReadBinFile(modelPath, mdlData, mdlSize);

    dclError ret = dclmdlQuerySizeFromMem(mdlData, mdlSize, &modelWorkSize_, &modelWeightSize_);
    if (ret != DCL_ERROR_NONE) {
        ERROR_LOG("query model failed, model file is %s, errorCode is %d",
            modelPath, static_cast<int32_t>(ret));
        return FAILED;
    }

    ret = dclmdlLoadFromMem(mdlData, mdlSize, &modelId_);
    if (ret != DCL_ERROR_NONE) {
        ERROR_LOG("load model from file failed, model file is %s, errorCode is %d",
            modelPath, static_cast<int32_t>(ret));
        return FAILED;
    }

    if(0 == aippFlag){
        ret = dclmdlSetAippDisable(modelId_);
        if (ret != DCL_ERROR_NONE) {
            ERROR_LOG("disable aipp failure, modelId %d errorCode is %d", modelId_, ret);
            return FAILED;
        }
        INFO_LOG("disable aipp success");
    }   

    loadFlag_ = true;
    ret =dclrtFree(mdlData);
    if(ret != DCL_ERROR_NONE){
        ERROR_LOG("free model device memory failed");
        return FAILED;
    }

    INFO_LOG("load model %s success", modelPath.c_str());
    return SUCCESS;
}

Result ModelProcess::CreateModelDesc()
{
    modelDesc_ = dclmdlCreateDesc();
    if (modelDesc_ == nullptr) {
        ERROR_LOG("create model description failed");
        return FAILED;
    }

    dclError ret = dclmdlGetDesc(modelDesc_, modelId_);
    if (ret != DCL_ERROR_NONE) {
        ERROR_LOG("get model description failed, modelId is %u, errorCode is %d",
            modelId_, static_cast<int32_t>(ret));
        return FAILED;
    }

    INFO_LOG("create model description success");

    return SUCCESS;
}

Result ModelProcess::init()
{
    // creat model descript
    Result ret;
    ret = CreateModelDesc();
    if (ret != SUCCESS) {
        ERROR_LOG("create model description failed");
        return FAILED;
    }
    return SUCCESS;
}

void ModelProcess::DestroyModelDesc()
{
    if (modelDesc_ != nullptr) {
        (void)dclmdlDestroyDesc(modelDesc_);
        modelDesc_ = nullptr;
    }
    INFO_LOG("destroy model description success");
}

void ModelProcess::DumpModelOutputResult(dclmdlDataset *output)
{
    stringstream ss;
    size_t outputNum = dclmdlGetDatasetNumBuffers(output);
    static int executeNum = 0;
    for (size_t i = 0; i < outputNum; ++i) {
        ss << "output" << ++executeNum << "_" << i << ".bin";
        string outputFileName = ss.str();
        FILE *outputFile = fopen(outputFileName.c_str(), "wb");
        if (outputFile != nullptr) {
            // get model output data
            dclDataBuffer *dataBuffer = dclmdlGetDatasetBuffer(output, i);
            void *data = dclGetDataBufferAddr(dataBuffer);
            uint32_t len = dclGetDataBufferSize(dataBuffer);
            fwrite(data, len, sizeof(char), outputFile);
            fclose(outputFile);
        } else {
            ERROR_LOG("create output file [%s] failed", outputFileName.c_str());
            return;
        }
    }

    INFO_LOG("dump data success");
    return;
}

Result ModelProcess::ExecuteAsync(uint32_t modelId, dclmdlDataset *input, dclmdlDataset *output, dclNnHandle* handle)
{
    dclError ret = dclmdlExecuteAsync(modelId, input, output, handle);
    {
        std::unique_lock<std::mutex> lck (mtx_handle_);
        ModelExcuteInfo info={*handle, input, output, false};
        ModelExcuteInfos.push_back(info);
    }
    return SUCCESS;
}


Result ModelProcess::ExecuteSync(uint32_t modelId, dclmdlDataset *input, dclmdlDataset *output) 
{
    dclError ret = dclmdlExecute(modelId, input, output);
    if (ret != DCL_ERROR_NONE) {
        ERROR_LOG("execute model sync failed, modelId is %u, errorCode is %d",
            modelId, static_cast<int32_t>(ret));
        return FAILED;
    }
    return SUCCESS;
}

void ModelProcess::UnloadModel()
{
    if (!loadFlag_) {
        WARN_LOG("no model had been loaded, unload failed");
        return;
    }

    dclError ret = dclmdlUnload(modelId_);
    if (ret != DCL_ERROR_NONE) {
        ERROR_LOG("unload model failed, modelId is %u, errorCode is %d",
            modelId_, static_cast<int32_t>(ret));
    }

    if (modelDesc_ != nullptr) {
        (void)dclmdlDestroyDesc(modelDesc_);
        modelDesc_ = nullptr;
    }

    if (modelWorkPtr_ != nullptr) {
        (void)dclrtFree(modelWorkPtr_);
        modelWorkPtr_ = nullptr;
        modelWorkSize_ = 0;
    }

    if (modelWeightPtr_ != nullptr) {
        (void)dclrtFree(modelWeightPtr_);
        modelWeightPtr_ = nullptr;
        modelWeightSize_ = 0;
    }

    loadFlag_ = false;
    INFO_LOG("unload model success, modelId is %u", modelId_);
    modelId_ = 0;
}
