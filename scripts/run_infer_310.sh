#!/bin/bash
# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===========================================================================
if [ $# != 3 ] && [ $# != 2 ]
then
    echo "Usage: bash run_infer_310.sh [MINDIR_PATH] [DEVICE_ID]"
    echo "OR"
    echo "Usage: bash run_infer_310.sh [CONFIG_PATH] [MINDIR_PATH] [DEVICE_ID]"
exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

device_id=0

if [ $# == 2 ]
then
  model=$(get_real_path $1)
  device_id=$2
  CONFIG_PATH=$"./config/yolov7/net/yolov7.yaml"
fi

if [ $# == 5 ]
then
  CONFIG_PATH=$(get_real_path $1)
  model=$(get_real_path $2)
  device_id=$3
fi

echo $model
echo $CONFIG_PATH

function compile_app()
{
    cd ascend310_infer || exit
    if [ -f "Makefile" ]; then
        make clean
    fi
    sh build.sh &> build.log

    if [ $? -ne 0 ]; then
        echo "compile app code failed"
        exit 1
    fi
    cd - || exit
}

function preprocess_data()
{
    if [ -d preprocess_Result ]; then
        rm -rf ./preprocess_Result
    fi
    mkdir preprocess_Result
    python preprocess.py --output_path=./preprocess_Result
}

function infer()
{
    if [ -d result_Files ]; then
        rm -rf ./result_Files
    fi
    if [ -d time_Result ]; then
        rm -rf ./time_Result
    fi
    mkdir result_Files
    mkdir time_Result
    ascend310_infer/out/main --model_path=$model --dataset_path=$data_path --device_id=$device_id &> infer.log

    if [ $? -ne 0 ]; then
        echo "execute inference failed"
        exit 1
    fi
}

function cal_acc()
{
    python postprocess.py --result_path=result_Files &> acc.log

    if [ $? -ne 0 ]; then
        echo "calculate accuracy failed"
        exit 1
    fi
}

preprocess_data
data_path=./preprocess_Result/img_data
compile_app
infer
cal_acc