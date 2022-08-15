/**
* @file device_process.h
*
* Copyright (C) 2020. intellif Technologies Co., Ltd. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/
#ifndef DEVICE_PROCESS_H_
#define DEVICE_PROCESS_H_

#include "utils.h"
#include "dcl.h"

class deviceProcess {
public:
    /**
    * @brief Constructor
    */
    deviceProcess();

    /**
    * @brief Destructor
    */
    virtual ~deviceProcess();

    /**
    * @brief init reousce
    * @return result
    */
    Result InitResource(const std::string &config_file);
public:
    //static dclrtContext context_;
    //static dclrtStream stream_;
private:
    void DestroyResource();

    int32_t deviceId_;

};

#endif // SAMPLE_PROCESS_H_
