/**
* @file model_process.h
*
* Copyright (C) 2020. intellif Technologies Co., Ltd. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/
#ifndef MODEL_PROCESS_H_
#define MODEL_PROCESS_H_

#include <iostream>
#include <vector>
#include "utils.h"
#include "dcl.h"
#include <mutex> 
#include <atomic>
#include "dcl_mdl.h"
#include <thread>

typedef void(*NNResultCallBack)(void *data, void* para, void* usr_data);
typedef struct ModelExcuteInfo{
    dclNnHandle handle;
    dclmdlDataset* input;
    dclmdlDataset* output;
    bool finished;
};
class ModelProcess {
public:
    /**
    * @brief Constructor
    */
    ModelProcess();

    /**
    * @brief Destructor
    */
    virtual ~ModelProcess();

    /**
    * @brief load model
    * @param [in] modelPath: model path
    * @return result
    */
    Result LoadModel(const std::string &modelPath, int aippFlag);

    /**
    * @brief unload model
    */
    void UnloadModel();

    /**
    * @brief int model process
    * @return result
    */
    Result init();

    /**
    * @brief model async execute
    * @return result
    */
    Result ExecuteAsync(uint32_t modelId, dclmdlDataset *input, dclmdlDataset *output, dclNnHandle* pNnHandle);


    void DumpModelOutputResult(dclmdlDataset *output);
    /**
    * @brief query output
    * @return result
    */
    //Result SetNNResultCallback();
    /**
    * @brief model async execute
    * @return result
    */
    Result ExecuteSync(uint32_t modelId, dclmdlDataset *input, dclmdlDataset *output);

    void PrintIOInfo(int aippFlag) {
        dclmdlIODims dims;

        if (!aippFlag) { // AIPP打开时，模型中的输入信息无效，由用户根据实际输入配置
            int inputNum = dclmdlGetNumInputs(modelDesc_);
            INFO_LOG("model input num=%d", inputNum);
            for (int i = 0; i < inputNum; i++) {
                dclmdlGetInputDims(modelDesc_, i, &dims);
                dclFormat format = dclmdlGetInputFormat(modelDesc_, i);
                dclDataType datatype = dclmdlGetInputDataType(modelDesc_, i);
                int d = dims.dimCount;
                INFO_LOG("input[%d] dim=%d, format=%d, datatype=%d", i, d, format, datatype);
                for (int j = 0; j < d; j++) {
                    INFO_LOG("shape[%d]=%lld", j, dims.dims[j]);
                }
            }
        }

        int outputNum = dclmdlGetNumOutputs(modelDesc_);
        INFO_LOG("model output num=%d", outputNum);
        for (int i = 0; i < outputNum; i++) {
            dclmdlGetOutputDims(modelDesc_, i, &dims);
            dclFormat format = dclmdlGetOutputFormat(modelDesc_, i);
            dclDataType datatype = dclmdlGetOutputDataType(modelDesc_, i);
            int d = dims.dimCount;
            INFO_LOG("output[%d] dim=%d, format=%d, datatype=%d", i, d, format, datatype);
            for (int j = 0; j < d; j++) {
                INFO_LOG("shape[%d]=%lld", j, dims.dims[j]);
            }
        }
    }


    uint32_t modelId_;
    dclmdlDesc *modelDesc_;
    std::mutex mtx_handle_; 
    std::vector<ModelExcuteInfo> ModelExcuteInfos;
    std::atomic<bool> run_flg_;
private:
    
    size_t modelWorkSize_; // model work memory buffer size
    size_t modelWeightSize_; // model weight memory buffer size
    void *modelWorkPtr_; // model work memory buffer
    void *modelWeightPtr_; // model weight memory buffer
    bool loadFlag_;  // model load flag 
    //dclrtStream stream_;

    void DestroyModelDesc();
    Result CreateModelDesc();
   
    std::thread *result_thd_;



};

#endif // MODEL_PROCESS_H_