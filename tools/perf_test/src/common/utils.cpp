/**
* @file utils.cpp
*
* Copyright (C) 2020. intellif Technologies Co., Ltd. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/
#include "utils.h"
#include <iostream>
#include <fstream>
#include <cstring>
#include <sys/stat.h>
#include "dcl.h"

extern bool g_isDevice;

Result Utils::SaveBinfile(const std::string &name, const char *ptr, size_t size){
    std::ofstream ofile(name.c_str(), std::ios::out|std::ios::binary);
    if (ofile.is_open()){
        ofile.write(ptr, size);
        ofile.close();
        return SUCCESS;
    }
    else{
        std::cout << "save bin file " << name.c_str() << " failure!" << std::endl;
        return FAILED;
    }
}    

Result Utils::ReadBinFile(const std::string &fileName, void *&inputBuff, uint32_t &fileSize)
{
    if (CheckPathIsFile(fileName) == FAILED) {
        ERROR_LOG("%s is not a file", fileName.c_str());
        return FAILED;
    }

    std::ifstream binFile(fileName, std::ifstream::binary);
    if (binFile.is_open() == false) {
        ERROR_LOG("open file %s failed", fileName.c_str());
        return FAILED;
    }

    binFile.seekg(0, binFile.end);
    uint32_t binFileBufferLen = binFile.tellg();
    if (binFileBufferLen == 0) {
        ERROR_LOG("binfile is empty, filename is %s", fileName.c_str());
        binFile.close();
        return FAILED;
    }
    binFile.seekg(0, binFile.beg);

    dclError ret = DCL_ERROR_NONE;
    ret = dclrtMalloc(&inputBuff, binFileBufferLen, DCL_MEM_MALLOC_NORMAL_ONLY);
    if (ret != DCL_ERROR_NONE) {
        ERROR_LOG("malloc device buffer failed. size is %u, errorCode is %d",
            binFileBufferLen, static_cast<int32_t>(ret));
        binFile.close();
        return FAILED;
    }
    binFile.read(static_cast<char *>(inputBuff), binFileBufferLen);
    binFile.close();
    fileSize = binFileBufferLen;
    return SUCCESS;
}

Result Utils::GetDeviceBufferOfFile(const std::string &fileName, void *&picDevBuffer, uint32_t &fileSize)
{
    void *inputBuff = nullptr;
    uint32_t inputBuffSize = 0;
    auto ret = Utils::ReadBinFile(fileName, inputBuff, inputBuffSize);
    if (ret != SUCCESS) {
        ERROR_LOG("read bin file failed, file name is %s", fileName.c_str());
        return FAILED;
    }
    picDevBuffer = inputBuff;
    fileSize = inputBuffSize;
    return SUCCESS;
}

Result Utils::CheckPathIsFile(const std::string &fileName)
{
#if defined(_MSC_VER)
    DWORD bRet = GetFileAttributes((LPCSTR)fileName.c_str());
    if (bRet == FILE_ATTRIBUTE_DIRECTORY) {
        ERROR_LOG("%s is not a file, please enter a file", fileName.c_str());
        return FAILED;
    }
#else
    struct stat sBuf;
    int fileStatus = stat(fileName.data(), &sBuf);
    if (fileStatus == -1) {
        ERROR_LOG("failed to get file");
        return FAILED;
    }
    if (S_ISREG(sBuf.st_mode) == 0) {
        ERROR_LOG("%s is not a file, please enter a file", fileName.c_str());
        return FAILED;
    }
#endif
    return SUCCESS;
}
