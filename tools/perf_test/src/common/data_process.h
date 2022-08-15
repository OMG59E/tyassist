/**
* @file data_process.h
*
* Copyright (C) 2020. intellif Technologies Co., Ltd. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/
#ifndef DATA_PROCESS_H_
#define DATA_PROCESS_H_

#include <iostream>
#include <vector>
#include <mutex>
#include <map>
#include "utils.h"
#include "dcl.h"

#ifdef CPU_SIMU
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif


using namespace std;

typedef struct _AippInfo {
    int width;
    int height;
    dclAippInputFormat format;
} AippInfo;

class DataProcess {
public:
    /**
    * @brief Constructor
    */
    DataProcess() {}

    /**
    * @brief Destructor
    */
    virtual ~DataProcess() {
        Destroy();
    }

    /**
    * @brief init memory pool, with AIPP Disabled
    * @param [in] modelDesc: model description
    */
    Result Init(vector<string>& testFiles, dclmdlDesc *modelDesc, uint32_t modelId, AippInfo *aippInfo = nullptr) {
        Result ret = SUCCESS;
        int testNum = testFiles.size();

        if (testNum != 1) {
            ERROR_LOG("DataProcess only support 1 input file, now is %d", testNum);
            return FAILED;
        }

        for (size_t i = 0; i < testNum; ++i) {
            uint32_t devBufferSize;
            void *picDevBuffer = nullptr;

            // 读取输入数据
            #ifdef CPU_SIMU
            cv::Mat img = cv::imread(testFiles[i]);
            INFO_LOG("imread pic %s, %d * %d", testFiles[i].c_str(), img.cols, img.rows);
            // Size dsize = Size(416, 416);
            // Mat img2 = Mat(dsize, CV_32S);
            // resize(img, img2, dsize, 0, 0, INTER_CUBIC);
            // resize(img, img2, dsize);
            Mat2RgbPlane(img, picDevBuffer, devBufferSize);

            if (aippInfo) {
                aippInfo->width = img.cols;
                aippInfo->height = img.rows;
            }
            // Utils::SaveBinfile(testFiles[i]+".416x416.rgb.plane.bin", (char*)picDevBuffer, devBufferSize);
            #else
            ret = Utils::GetDeviceBufferOfFile(testFiles[i], picDevBuffer, devBufferSize);
            if (ret != SUCCESS) {
                ERROR_LOG("get pic device buffer failed, index is %zu", i);
                return FAILED;
            }
            #endif

            // 创建模型输入空间
            ret = CreateInput(picDevBuffer, devBufferSize, input_, modelDesc, modelId, aippInfo);
            if (ret != SUCCESS) {
                ERROR_LOG("execute CreateInput failed");
                dclrtFree(picDevBuffer);
                return FAILED;
            }

            // 创建模型输出空间
            ret = CreateOutput(output_, modelDesc);
            if (ret != SUCCESS) {
                ERROR_LOG("execute CreateOutput failed");
                return FAILED;
            }
        }
        return SUCCESS;
    }

    /**
    * @brief destroy memory pool
    */
    void Destroy() {
        if (input_) {
            for (size_t i = 0; i < dclmdlGetDatasetNumBuffers(input_); ++i) {
                dclDataBuffer* dataBuffer = dclmdlGetDatasetBuffer(input_, i);
                void *data = dclGetDataBufferAddr(dataBuffer);
                (void)dclrtFree(data);
                (void)dclDestroyDataBuffer(dataBuffer);
            }
            (void)dclmdlDestroyDataset(input_);
        }

        if (output_) {
            for (size_t i = 0; i < dclmdlGetDatasetNumBuffers(output_); ++i) {
                dclDataBuffer* dataBuffer = dclmdlGetDatasetBuffer(output_, i);
                void *data = dclGetDataBufferAddr(dataBuffer);
                (void)dclrtFree(data);
                (void)dclDestroyDataBuffer(dataBuffer);
            }
            (void)dclmdlDestroyDataset(output_);
        }
    }

    /**
    * @brief create model input
    * @param [in] inputDataBuffer: input buffer
    * @param [in] bufferSize: input buffer size
    * @param [out] input: input dataset
    * @param [in] modelDesc: model description
    * @return result
    */
    Result CreateInput(void *inputDataBuffer, size_t bufferSize, dclmdlDataset *&input, dclmdlDesc *modelDesc, uint32_t modelId, AippInfo *aippInfo = nullptr) {
        if (modelDesc == nullptr) {
            ERROR_LOG("no model description, create input failed");
            return FAILED;
        }

        dclmdlInputAippType type;
        size_t attachIndex;
        dclError ret = dclmdlGetAippType(modelId, 0, &type, &attachIndex);
        if(DCL_ERROR_NONE != ret){
            ERROR_LOG("model get aipp type failed");
            return FAILED;       
        }

        input = dclmdlCreateDataset();
        if (input == nullptr) {
            ERROR_LOG("can't create dataset, create input failed");
            return FAILED;
        }

        dclDataBuffer* inputData = dclCreateDataBuffer(inputDataBuffer, bufferSize);
        if (inputData == nullptr) {
            ERROR_LOG("can't create data buffer, create input failed");
            (void)dclmdlDestroyDataset(input);
            input = nullptr;
            return FAILED;
        }

        ret = dclmdlAddDatasetBuffer(input, inputData);
        if (ret != DCL_ERROR_NONE) {
            ERROR_LOG("add input dataset buffer failed, errorCode is %d", static_cast<int32_t>(ret));
            (void)dclDestroyDataBuffer(inputData);
            inputData = nullptr;
            (void)dclmdlDestroyDataset(input);
            input = nullptr;
            return FAILED;
        } 

        if (aippInfo) {
            // 增加AIPP输入
            dclAippInputFormat format = aippInfo->format;
            auto aippSize = dclmdlGetInputSizeByIndex(modelDesc, attachIndex);
            void *aippData = nullptr;
            dclrtMalloc(&aippData, aippSize, DCL_MEM_MALLOC_NORMAL_ONLY);
            dclDataBuffer *aippBuf = dclCreateDataBuffer(aippData, aippSize);
            dclmdlAddDatasetBuffer(input, aippBuf);

            dclmdlAIPP *aipp = dclmdlCreateAIPP();
            dclmdlSetAIPPInputFormat(aipp, format);
            dclmdlSetAIPPSrcImageSize(aipp, aippInfo->width, aippInfo->height);
            dclmdlSetAIPPCropParams(aipp, 0, 0, 0, aippInfo->width, aippInfo->height);
            dclmdlSetAIPPByInputIndex(modelId, input, 0, aipp);
            dclmdlDestroyAIPP(aipp);  
        }

        INFO_LOG("create input success");
        return SUCCESS;
    }

    /**
    * @brief create output buffer
    * @param [out] output: output dataset
    * @param [in] modelDesc: model description
    * @return result
    */
    Result CreateOutput(dclmdlDataset *&output, dclmdlDesc *modelDesc) {
        if (modelDesc == nullptr) {
            ERROR_LOG("no model description, create ouput failed");
            return FAILED;
        }
        output = dclmdlCreateDataset();
        if (output == nullptr) {
            ERROR_LOG("can't create dataset, create output failed");
            return FAILED;
        }

        size_t outputSize = dclmdlGetNumOutputs(modelDesc);
        for (size_t i = 0; i < outputSize; ++i) {
            size_t modelOutputSize = dclmdlGetOutputSizeByIndex(modelDesc, i);

            void *outputBuffer = nullptr;
            dclError ret = dclrtMalloc(&outputBuffer, modelOutputSize, DCL_MEM_MALLOC_NORMAL_ONLY);
            if (ret != DCL_ERROR_NONE) {
                ERROR_LOG("can't malloc buffer, create output failed, size is %zu, errorCode is %d",
                    modelOutputSize, static_cast<int32_t>(ret));
                (void)dclmdlDestroyDataset(output);
                return FAILED;
            }

            dclDataBuffer* outputData = dclCreateDataBuffer(outputBuffer, modelOutputSize);
            if (outputData == nullptr) {
                ERROR_LOG("can't create data buffer, create output failed");
                (void)dclmdlDestroyDataset(output);
                (void)dclrtFree(outputBuffer);
                output = nullptr;
                return FAILED;
            }
            ret = dclmdlAddDatasetBuffer(output, outputData);
            if (ret != DCL_ERROR_NONE) {
                ERROR_LOG("can't add data buffer, create output failed");
                (void)dclmdlDestroyDataset(output);
                (void)dclrtFree(outputBuffer);
                (void)dclDestroyDataBuffer(outputData);
                output = nullptr;
                return FAILED;
            }
            INFO_LOG("add output index %zu size %zu", i, modelOutputSize);
        }

        INFO_LOG("create output success");
        return SUCCESS;
    }

    dclmdlDataset* GetInputDataSet() {
        return input_;
    }

    dclmdlDataset* GetOutputDataSet() {
        return output_;
    }

    #ifdef CPU_SIMU
    static void Mat2RgbPlane(cv::Mat &img, void *&devPtr, uint32_t &size) {
        size = 3 * img.cols * img.rows;
        dclrtMalloc(&devPtr, size, DCL_MEM_MALLOC_NORMAL_ONLY);
        
        char *p=(char*)devPtr;
        for(uint32_t ch = 0; ch < 3; ch++)    // BGR ==> RGB_PLANE
        {
            uint32_t ch_bit = ch;
            //==>DE_PIX_FMT_RGB888_PLANE
            if(ch == 0) ch_bit = 2;
            if(ch == 2) ch_bit = 0;

            for(int32_t r = 0; r < img.rows; r++)
            {
                for(int32_t c = 0; c < img.cols; c++)
                {
                    p[ch * img.rows * img.cols + r * img.cols + c] = img.data[r * img.cols * 3 + c * 3 + ch_bit];
                }
            }
        }
    }
    #endif

protected:

    dclmdlDataset* input_ = nullptr;
    dclmdlDataset* output_ = nullptr;

};

#endif // MEMORY_POOL_H_
