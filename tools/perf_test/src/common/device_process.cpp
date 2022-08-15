/**
* @file sample_process.cpp
*
* Copyright (C) 2020. intellif Technologies Co., Ltd. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/
#include "device_process.h"
#include <iostream>
#include <thread>
#include <sstream>
#include "dcl.h"
#include "utils.h"
#include "model_process.h"

using namespace std;

deviceProcess::deviceProcess() :deviceId_(0)
{
}

deviceProcess::~deviceProcess()
{
    DestroyResource();
}

Result deviceProcess::InitResource(const string &config_file)
{
    dclError ret = dclrtMemInit(0); //Non-Cached
    if (ret != DCL_ERROR_NONE) {
        ERROR_LOG("dcl mem init failed, errorCode = %d", static_cast<int32_t>(ret));
        return FAILED;
    }
    INFO_LOG("dcl mem init success");

    ret = dclInit(config_file.c_str());
    if (ret != DCL_ERROR_NONE) {
        ERROR_LOG("dcl init failed, errorCode = %d", static_cast<int32_t>(ret));
        return FAILED;
    }
    INFO_LOG("dcl init success");
    return SUCCESS;
}


void deviceProcess::DestroyResource()
{
    dclError ret;
    ret = dclFinalize();
    if (ret != DCL_ERROR_NONE) {
        ERROR_LOG("finalize dcl failed, errorCode = %d", static_cast<int32_t>(ret));
        return;
    }
    INFO_LOG("end to finalize dcl");

    ret = dclrtMemDeinit();
    if (ret != DCL_ERROR_NONE) {
        ERROR_LOG("memory de-init failed, errorCode = %d", static_cast<int32_t>(ret));
        return;
    }    
    INFO_LOG("memory de-init done");
}
