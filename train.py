#  This file is part of PolyLUT-Add.
#
#  PolyLUT-Add is a derivative work based on PolyLUT,
#  which is licensed under the Apache License 2.0.

#  Copyright (C) 2021 Xilinx, Inc
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import os
from argparse import ArgumentParser
from functools import reduce
import random

import numpy as np
import wandb
import random
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import JetSubstructureDataset
from models import JetSubstructureNeqModel, JetSubstructureNeqModel_add2
from polylut.nn import generate_truth_tables
from polylut.nn import SparseLinearNeq
from polylut.nn import SparseLinearNeq_add2
from polylut.nn import Adder2
import subprocess
import re
from functools import lru_cache
import tempfile
from concurrent.futures import ThreadPoolExecutor
import hashlib
import time
from multiprocessing import Pool
from collections import Counter
from itertools import islice
import threading


configs = {
    "jsc-m-lite-add2": {
        "hidden_layers": [64, 32],
        "input_bitwidth": 3,
        "hidden_bitwidth": 3,
        "output_bitwidth": 3,
        "input_fanin": 2,
        "degree": 3,
        "hidden_fanin": 2,
        "output_fanin": 2,
        "weight_decay": 0,
        "batch_size": 1024,
        "epochs": 500,
        "learning_rate": 1e-3,
        "seed": 1697,
        "checkpoint": None,
    },
    "jsc-xl-add2": {
        "hidden_layers": [128, 64, 64, 64],
        "input_bitwidth": 7,
        "hidden_bitwidth": 5,
        "output_bitwidth": 5,
        "input_fanin": 1,
        "degree": 3, 
        "hidden_fanin": 2,
        "output_fanin": 2,
        "weight_decay": 0,
        "batch_size": 1024,
        "epochs": 1000,
        "learning_rate":1e-2,
        "seed": 1234,
        "checkpoint": None,
    },
}

# A dictionary, so we can set some defaults if necessary
model_config = {
    "hidden_layers": None,
    "input_bitwidth": None,
    "hidden_bitwidth": None,
    "output_bitwidth": None,
    "input_fanin": None,
    "degree": None,
    "hidden_fanin": None,
    "output_fanin": None,
}

training_config = {
    "weight_decay": None,
    "batch_size": None,
    "epochs": None,
    "learning_rate": None,
    "seed": None,
}

dataset_config = {
    "dataset_file": None,
    "dataset_config": None,
}

other_options = {
    "cuda": None,
    "log_dir": None,
    "checkpoint": None,
    "device": 1,
}

def create_hook(w_list):
    def modify_grad(grad):
        for i in range(len(grad)):
            grad[i] *= w_list[i]
        return grad
    return modify_grad

def generate_count_map(n):
    if n < 1:
        raise ValueError("n must be at least 1")

    min_val = - (1 << (n - 1))
    max_val = (1 << (n - 1)) - 1

    lookup_table = {}

    for decimal in range(min_val, max_val + 1):
        if decimal < 0:
            binary = bin(decimal & ((1 << n) - 1))[2:].zfill(n)
            binary = '1' + binary[1:]
        else:
            binary = bin(decimal)[2:].zfill(n)
            binary = '0' + binary[1:]

        binary_list = list(binary)
        binary_list[0] = '1' if binary_list[0] == '0' else '0'
        binary = ''.join(binary_list)
        binary_tuple = tuple(int(bit) for bit in binary)
        lookup_table[decimal] = binary_tuple

    return lookup_table

def generate_adder_count_map(N):
    count_map_liner = {}

    for i in range(2 ** N):
        bin_tuple = tuple(int(x) for x in f'{i:0{N}b}')
        count_map_liner[i] = bin_tuple

    for i in range(2 ** (N - 1)):
        neg_value = -(i + 1)
        bin_tuple = tuple(int(x) for x in f'{(2 ** N + neg_value):0{N}b}')
        count_map_liner[neg_value] = bin_tuple

    return count_map_liner

def list_to_hex_lsb(tt_list):
    """
    输入：一个 0/1 组成的真值表列表（MSB-first）
    输出：LSB-first 的 HEX 字符串，用于 ABC 的 testnpn
    """
    assert all(x in [0, 1] for x in tt_list), "真值表必须是 0/1 构成"
    bin_str = ''.join(str(x) for x in reversed(tt_list))  # 转为 LSB-first 的字符串
    hex_str = hex(int(bin_str, 2))[2:]  # 转为十六进制，去掉 '0x'
    hex_str = hex_str.zfill(len(tt_list) // 4)
    return hex_str.lower()

def extract_single_npn_class(file_path):
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    # 精确匹配以 0x 开头、后跟至少 1 位十六进制字符，末尾必须是空格或换行
    match = re.search(r'0x[0-9a-fA-F]+\b', content)
    
    if match:
        return match.group(0)
    else:
        return None

def hex_to_binary_list(hex_str):
    # 去掉可能的空格或换行
    hex_str = hex_str.strip().lower().replace("_", "")
    bin_str = bin(int(hex_str, 16))[2:]
    # 补齐前导0
    length = len(hex_str) * 4
    bin_str = bin_str.zfill(length)
    return [int(b) for b in bin_str]

def generate_blif_from_tt(tt, output_dir=".", model_name="truth_table_example"):
    if not all(v in (0, 1) for v in tt):
        raise ValueError("error: only 0 and 1")
    num_entries = len(tt)
    if num_entries == 0 or (num_entries & (num_entries - 1)) != 0:
        raise ValueError("error: length")

    num_inputs = int(math.log2(num_entries))
    inputs = [chr(ord('a') + i) for i in range(num_inputs)]
    output = "out"

    os.makedirs(output_dir, exist_ok=True)
    blif_path = os.path.join(output_dir, f"{model_name}.blif")

    with open(blif_path, "w") as f:
        f.write(f".model {model_name}\n")
        f.write(f".inputs {' '.join(inputs)}\n")
        f.write(f".outputs {output}\n")
        f.write(f".names {' '.join(inputs)} {output}\n")
        for i, val in enumerate(tt):
            if val == 1:
                bin_str = format(i, f"0{num_inputs}b")
                f.write(f"{bin_str} 1\n")
        f.write(".end\n")

    return blif_path

def expand_dash_pattern(pattern):
    """将含有'-'的模式展开为所有可能的具体输入"""
    if '-' not in pattern:
        return [pattern]
    else:
        return expand_dash_pattern(pattern.replace('-', '0', 1)) + expand_dash_pattern(pattern.replace('-', '1', 1))

def extract_lut_truth_tables(blif_path):
    truth_tables = []
    if not os.path.exists(blif_path):
        truth_tables.append([0])
        return truth_tables
    with open(blif_path, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith(".names"):
            # 提取输入变量数（除去最后一个输出）
            parts = line.split()
            num_inputs = len(parts) - 2
            table_size = 2 ** num_inputs
            i += 1

            # 找到对应的 subckt 注释
            while i < len(lines) and not lines[i].startswith("#subckt dslut6:"):
                i += 1
            if i >= len(lines):
                break

            line = lines[i].strip()
            default_val = 1 if "False" in line else 0  # False 表示这些组合是0，其余是1
            override_val = 1 - default_val
            table = [default_val] * table_size

            i += 1
            while i < len(lines):
                l = lines[i].strip()
                if not l or l.startswith(".") or l.startswith("#"):
                    break
                if len(l.split()) == 2:
                    pattern, val = l.split()
                    for expanded in expand_dash_pattern(pattern):
                        idx = int(expanded, 2)
                        table[idx] = int(val)
                i += 1

            truth_tables.append(table)
        else:
            i += 1
    return truth_tables

def process_npn_group(tt_block, idx=None):
    # 生成唯一模型名，避免线程间冲突
    model_name = f"from_hex_tt_{os.getpid()}_{idx}" if idx is not None else "from_hex_tt"

    generate_blif_from_tt(tt_block, output_dir="output_blif", model_name=model_name)

    abc_cmd = f'{ABC_PATH} -c "read_blif output_blif/{model_name}.blif; if -n -K 6; write_blif output_blif/{model_name}_out.blif;"'
    subprocess.run(abc_cmd, shell=True, capture_output=True, text=True)

    tt_list = extract_lut_truth_tables(f"output_blif/{model_name}_out.blif")
    
    local_counts = {}
    for tt in tt_list:
        hex_result = list_to_hex_lsb(tt)
        try:
            npn_hex = cached_npn_class(hex_result)
            local_counts[npn_hex] = local_counts.get(npn_hex, 0) + 1
        except Exception as e:
            print(f"[Thread-{idx}] Error: {e}")
    
    return local_counts

ABC_PATH = "/home/mychen/abc/abc"
hash_module = {}

@lru_cache(maxsize=10240)
def cached_npn_class(hex_result: str) -> str:
    # 使用临时文件替代硬编码文件名，避免并发冲突
    with tempfile.NamedTemporaryFile("w+", delete=True, suffix=".txt") as f:
        f.write(f"{hex_result}\n")
        f.flush()
        abc_cmd = f'{ABC_PATH} -c "testnpn -A 8 -v {f.name}"'
        result = subprocess.run(abc_cmd, shell=True, capture_output=True, text=True, encoding='utf-8', errors='ignore')
    content = result.stdout + result.stderr
    # 直接从 stdout 提取 npn 结果，避免再次读文件
    match = re.search(r'0x[0-9a-fA-F]+\b', content)
    if match:
        return match.group(0)
    else:
        print(hex_result)
        print(content)
        raise RuntimeError(f"Failed to parse NPN class from ABC output:\n{content}")

def get_sub_lists(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def extract_npn_lists(temp_list, npn_num, cmap):
    npn_list = [[] for _ in range(npn_num)]
    for temp_num in temp_list:
        if isinstance(temp_num, torch.Tensor):
            temp_num = temp_num.item()
        if temp_num in cmap:
            for i in range(npn_num):
                npn_list[i].append(cmap[temp_num][i])
        else:
            print(f"Error in test DSLUT: {temp_num}")
    return npn_list


def process_npn_class_parallel(tt_block, abc_path, idx, output_dir="output_blif"):
    model_name = f"from_hex_tt_{os.getpid()}_{idx}"
    generate_blif_from_tt(tt_block, output_dir=output_dir, model_name=model_name)
    abc_cmd = f'{abc_path} -c "read_blif {output_dir}/{model_name}.blif; if -n -K 6; write_blif {output_dir}/{model_name}_out.blif;"'
    subprocess.run(abc_cmd, shell=True, capture_output=True, text=True)
    tt_list = extract_lut_truth_tables(f"{output_dir}/{model_name}_out.blif")
    hex_list = []
    for tt in tt_list:
        npn_hex = cached_npn_class(list_to_hex_lsb(tt))
        hex_list.append(npn_hex)

    local_counts = Counter(hex_list)

    return [dict(local_counts),hex_list]


output_dir="output_blif"

'''
def parallel_process_module(module, abc_path, count_map_liner, npn_number, output_dir="output_blif", max_workers=8):
    os.makedirs(output_dir, exist_ok=True)
    hash_func = {}
    hex_map = {}  # 每个 func_id -> list of hex strings

    task_list = []
    task_id = 0

    # === Step 1: 写出所有 .blif 文件，并建立 func_id -> path 映射 ===
    for func in module.neuron_truth_tables:
        temp_list = func[3]
        npn_lists = extract_npn_lists(temp_list, npn_number, count_map_liner)
        func_id = id(func[3])
        hex_map[func_id] = []

        for i, npn_block in enumerate(npn_lists):
            model_name = f"from_tt_{task_id}"
            blif_path = generate_blif_from_tt(npn_block, output_dir=output_dir, model_name=model_name)
            task_list.append((model_name, func_id))
            task_id += 1

    # === Step 2: 写 runcmd.abc 脚本，批处理 if -K 6 ===
    abc_script_path = os.path.join(output_dir, "runcmd.abc")
    with open(abc_script_path, "w") as f:
        for model_name, _ in task_list:
            f.write(f"read_blif {output_dir}/{model_name}.blif\n")
            f.write("if -n -K 6\n")
            f.write(f"write_blif {output_dir}/{model_name}_out.blif\n")
        f.write("quit\n")

    # === Step 3: 调用 ABC 一次性执行所有任务 ===
    subprocess.run(f'{abc_path} -f {abc_script_path}', shell=True, capture_output=True, text=True)

    # === Step 4: 并行提取所有 NPN 类别 ===

    with Pool(max_workers) as pool:
        results = pool.map(extract_npn_from_blif, task_list)

    # === Step 5: 汇总每个 hex 的出现次数，以及每个神经元的 hex 列表 ===
    for result in results:
        for npn_hex, count in result[0].items():
            hash_func[npn_hex] = hash_func.get(npn_hex, 0) + count
    
    for future_list, func in zip(get_sub_lists(results, npn_number), module.neuron_truth_tables):
        hash_module[id(func[3])] = list(map(lambda element: element[1], future_list))

    return hash_func

'''
# 并行处理模块
def parallel_process_module(module, abc_path, count_map_liner, npn_number, output_dir="output_blif", max_workers=8):
    hash_func = Counter()
    task_list = []
    task_id = 0

    for func in module.neuron_truth_tables:
        for idx in [3, 5]:
            temp_list = func[idx]
            npn_lists = extract_npn_lists(temp_list, npn_number, count_map_liner)
            for npn_block in npn_lists:
                task_list.append((npn_block, abc_path,task_id))
                task_id += 1

    with Pool(max_workers) as pool:
        results = pool.starmap(process_npn_class_parallel, task_list)

    for result in results:
        hash_func.update(result[0])

    #iter_results = iter(results)
    for future_list, func in zip(get_sub_lists(results, 2 * npn_number), module.neuron_truth_tables):
        hash_module[id(func[3])] = list(map(lambda element: element[1], future_list[0:npn_number - 1]))
        hash_module[id(func[5])] = list(map(lambda element: element[1], future_list[npn_number:2 * npn_number - 1]))

    return dict(hash_func)

def train(model, datasets, train_cfg, options, fre):
    # Create data loaders for training and inference:
    train_loader = DataLoader(
        datasets["train"], batch_size=train_cfg["batch_size"], shuffle=True
    )
    val_loader = DataLoader(
        datasets["valid"], batch_size=train_cfg["batch_size"], shuffle=False
    )
    test_loader = DataLoader(
        datasets["test"], batch_size=train_cfg["batch_size"], shuffle=False
    )

    # Configure optimizer
    weight_decay = train_cfg["weight_decay"]
    decay_exclusions = [
        "bn",
        "bias",
        "learned_value",
    ]  # Make a list of parameters name fragments which will ignore weight decay TODO: make this list part of the train_cfg
    decay_params = []
    no_decay_params = []
    for pname, params in model.named_parameters():
        if params.requires_grad:
            if reduce(
                lambda a, b: a or b, map(lambda x: x in pname, decay_exclusions)
            ):  # check if the current label should be excluded from weight decay
                # print("Disabling weight decay for %s" % (pname))
                no_decay_params.append(params)
            else:
                # print("Enabling weight decay for %s" % (pname))
                decay_params.append(params)
        # else:
        # print("Ignoring %s" % (pname))
    params = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    optimizer = optim.AdamW(
        params,
        lr=train_cfg["learning_rate"],
        betas=(0.5, 0.999),
        weight_decay=weight_decay,
    )

    # Configure scheduler
    steps = len(train_loader)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=steps * 100, T_mult=1
    )

    # Configure criterion
    criterion = nn.CrossEntropyLoss()

    # Push the model to the GPU, if necessary
    if options["cuda"]:
        model.cuda()

    # Main training loop
    maxAcc = 0.0
    bestTestAcc = 0.0
    bestAcc = 0.0
    bestValAcc = 0.01
    num_epochs = train_cfg["epochs"]
    hash_hook = {}
    with open('BestAccuracy.txt','w') as f:
        f.write("Begin train\n")
    with open('ValAccuracy.txt','w') as f:
        f.write("Begin train\n")
    with open('TestAccuracy.txt','w') as f:
        f.write("Begin train\n")
    npn_count = 4
    number_npn = []
    module_list = []
    for name, module in model.named_modules():
        if isinstance(module, SparseLinearNeq_add2):
            module_list.append(module)
    count_map = generate_adder_count_map(2)
    count_map_liner = generate_count_map(6)
    nnn = 0
    for epoch in range(0, num_epochs):
        # Train for this epoch
        start_time = time.time()
        nnn+=1
        model.train()
        accLoss = 0.0
        correct = 0
        npn_count += 1
        for batch_idx, (data, target) in enumerate(train_loader):
            if options["cuda"]:
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, torch.max(target, 1)[1])
            pred = output.detach().max(1, keepdim=True)[1]
            target_label = torch.max(target.detach(), 1, keepdim=True)[1]
            curCorrect = pred.eq(target_label).long().sum()
            curAcc = 100.0 * curCorrect / len(data)
            correct += curCorrect
            accLoss += loss.detach() * len(data)
            loss.backward()
            optimizer.step()
            scheduler.step()
        accLoss /= len(train_loader.dataset)
        with open('Loss.txt','w') as f:
            f.write(str(accLoss.item()))
        accuracy = 100.0 * correct / len(train_loader.dataset)
        with open('Accuracy.txt','w') as f:
            f.write(str(accuracy.item()))
        val_accuracy = test(model, val_loader, options["cuda"])
        with open('ValAccuracy.txt','a') as f:
            f.write(str(val_accuracy))
            f.write("\n")
        use_DSLUT = True
        # test for DSLUT
        if use_DSLUT and npn_count == 5:
            npn_count = 0
            number_npn = []
            #checkpoint = torch.load("./best_accuracy.pth")
            #model.load_state_dict(checkpoint['model_dict'])
            print("test for DSLUT")
            start_time = time.time()
            generate_truth_tables(model, verbose=False)
            for module in module_list:
                local_hash_func = parallel_process_module(
                    module=module,
                    abc_path=ABC_PATH,
                    #count_map=count_map_liner,
                    count_map_liner=count_map_liner,
                    npn_number=6,
                    output_dir="output_blif",
                    max_workers=30
                )
                #ahead_module = id(module)
                #hash_list[id(module)] = local_hash_func
                local_hash = local_hash_func
                local_npn = len(local_hash)
                total_count = sum(local_hash.values())
                ave_count = total_count / local_npn
                threshold = 0
                test_num = len([key for key, value in local_hash.items() if value > threshold])
                number_npn.append(test_num)
                #continue
                feedback_list_1 = []
                feedback_list_2 = []
                for func in module.neuron_truth_tables:
                    npn_list_1 = hash_module[id(func[3])]
                    npn_list_2 = hash_module[id(func[5])]
                    feedback = 0.25
                    for npn_list in npn_list_1:
                        for npn in npn_list:
                            if ave_count / local_hash[npn] < 1:
                                continue
                            feedback += (ave_count / local_hash[npn]) ** 6
                    feedback_list_1.append(feedback)
                    feedback = 0.25
                    for npn_list in npn_list_2:
                        for npn in npn_list:
                            if ave_count / local_hash[npn] < 1:
                                continue
                            feedback += (ave_count / local_hash[npn]) ** 6
                    feedback_list_2.append(feedback)
                if id(module.fc1) in hash_hook:
                    hash_hook[id(module.fc1)].remove()
                hook_handle = module.fc1.weight.register_hook(create_hook(feedback_list_1))
                hash_hook[id(module.fc1)] = hook_handle
                if id(module.fc2) in hash_hook:
                    hash_hook[id(module.fc2)].remove()
                hook_handle = module.fc2.weight.register_hook(create_hook(feedback_list_2))
                hash_hook[id(module.fc2)] = hook_handle
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Elapsed time: {elapsed_time} seconds")
            print("test end")
        else:
            hash_func = {}
        test_accuracy = test(model, test_loader, options["cuda"])
        print(number_npn)
        print(nnn)
        print(sum(number_npn))
        print(test_accuracy)
        with open('TestAccuracy.txt','a') as f:
            f.write(str(test_accuracy))
            f.write("\n")
        if test_accuracy>bestTestAcc:
            bestAcc = accuracy.item()
            bestTestAcc = test_accuracy
            bestValAcc = val_accuracy
            with open('BestAccuracy.txt','a') as f:
                f.write("BestAcc:")
                f.write(str(bestAcc))
                f.write("BestTestAcc:")
                f.write(str(bestTestAcc))
                f.write("BestValAcc:")
                f.write(str(bestValAcc))
                f.write("BestNPNlist:")
                f.write(str(number_npn))
                f.write("BestNPNnum:")
                f.write(str(sum(number_npn)))
                f.write("\n")
        modelSave = {
            "model_dict": model.state_dict(),
            "optim_dict": optimizer.state_dict(),
            "val_accuracy": val_accuracy,
            "test_accuracy": test_accuracy,
            "epoch": epoch,
        }
        torch.save(modelSave, "test_fan_" + str(config["output_fanin"]) + options["log_dir"] + "/checkpoint.pth")
        if maxAcc < test_accuracy:
            torch.save(modelSave, "test_fan_" + str(config["output_fanin"]) + options["log_dir"] + "/best_accuracy.pth")
            maxAcc = test_accuracy
        wandb.log(
            {
                "Train Acc (%)": accuracy.detach().cpu().numpy(),
                "Train Loss(%)": accLoss.detach().cpu().numpy(),
                "Test Acc (%)": test_accuracy,
                "Valid Acc(%)": val_accuracy,
            }
        )
        end_time = time.time()
        elapsed_time = end_time - start_time
        #print(f"Elapsed time: {elapsed_time} seconds")

def test(model, dataset_loader, cuda):
    model.eval()
    correct = 0
    accLoss = 0.0
    for batch_idx, (data, target) in enumerate(dataset_loader):
        if cuda:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        pred = output.detach().max(1, keepdim=True)[1]
        target_label = torch.max(target.detach(), 1, keepdim=True)[1]
        curCorrect = pred.eq(target_label).long().sum()
        curAcc = 100.0 * curCorrect / len(data)
        correct += curCorrect
    accuracy = 100 * float(correct) / len(dataset_loader.dataset)
    return accuracy


if __name__ == "__main__":
    parser = ArgumentParser(description="PolyLUT Example")
    parser.add_argument(
        "--arch",
        type=str,
        choices=configs.keys(),
        default="jsc-m-lite",
        metavar="",
        help="Specific the neural network model to use (default: %(default)s)",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=None,
        metavar="",
        help="Weight decay (default: %(default)s)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        metavar="",
        help="Batch size for training (default: %(default)s)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        metavar="",
        help="Number of epochs to train (default: %(default)s)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        metavar="",
        help="Initial learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--cuda",
        action="store_true",
        default=True,
        help="Train on a GPU (default: %(default)s)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        metavar="",
        help="Seed to use for RNG (default: %(default)s)",
    )
    parser.add_argument(
        "--input_bitwidth",
        type=int,
        default=None,
        metavar="",
        help="Bitwidth to use at the input (default: %(default)s)",
    )
    parser.add_argument(
        "--hidden_bitwidth",
        type=int,
        default=None,
        metavar="",
        help="Bitwidth to use for activations in hidden layers (default: %(default)s)",
    )
    parser.add_argument(
        "--output_bitwidth",
        type=int,
        default=None,
        metavar="",
        help="Bitwidth to use at the output (default: %(default)s)",
    )
    parser.add_argument(
        "--input_fanin",
        type=int,
        default=None,
        metavar="",
        help="Fanin to use at the input (default: %(default)s)",
    )
    parser.add_argument(
        "--degree",
        type=int,
        default=None,
        metavar="",
        help="Degree to use for polynomials (default: %(default)s)",
    )
    parser.add_argument(
        "--hidden_fanin",
        type=int,
        default=None,
        metavar="",
        help="Fanin to use for the hidden layers (default: %(default)s)",
    )
    parser.add_argument(
        "--output_fanin",
        type=int,
        default=None,
        metavar="",
        help="Fanin to use at the output (default: %(default)s)",
    )
    parser.add_argument(
        "--hidden_layers",
        nargs="+",
        type=int,
        default=None,
        metavar="",
        help="A list of hidden layer neuron sizes (default: %(default)s)",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="0",
        help="A location to store the log output of the training run and the output model (default: %(default)s)",
    )
    parser.add_argument(
        "--dataset_file",
        type=str,
        default="data/processed-pythia82-lhc13-all-pt1-50k-r1_h022_e0175_t220_nonu_truth.z",
        metavar="",
        help="The file to use as the dataset input (default: %(default)s)",
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default="config/yaml_IP_OP_config.yml",
        metavar="",
        help="The file to use to configure the input dataset (default: %(default)s)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        metavar="",
        help="Retrain the model from a previous checkpoint (default: %(default)s)",
    )
    parser.add_argument(
        "--device", 
        type=int, 
        default=0, 
        metavar="", 
        help="Device_id for GPU",
    )
    args = parser.parse_args()
    defaults = configs[args.arch]
    options = vars(args)
    del options["arch"]
    config = {}
    for k in options.keys():
        config[k] = (
            options[k] if options[k] is not None else defaults[k]
        )  # Override defaults, if specified.

    if not os.path.exists("test_fan_" + str(config["output_fanin"]) + config["log_dir"]):
        os.makedirs("test_fan_" + str(config["output_fanin"]) + config["log_dir"])

    # Split up configuration options to be more understandable
    model_cfg = {}
    for k in model_config.keys():
        model_cfg[k] = config[k]
    train_cfg = {}
    for k in training_config.keys():
        train_cfg[k] = config[k]
    dataset_cfg = {}
    for k in dataset_config.keys():
        dataset_cfg[k] = config[k]
    options_cfg = {}
    for k in other_options.keys():
        options_cfg[k] = config[k]

    # Set random seeds
    random.seed(train_cfg["seed"])
    np.random.seed(train_cfg["seed"])
    torch.manual_seed(train_cfg["seed"])
    os.environ["PYTHONHASHSEED"] = str(train_cfg["seed"])
    if options["cuda"]:
        torch.cuda.manual_seed_all(train_cfg["seed"])
        torch.backends.cudnn.deterministic = True
        torch.cuda.set_device(options_cfg["device"])

    # Fetch the datasets
    dataset = {}
    dataset["train"] = JetSubstructureDataset(
        dataset_cfg["dataset_file"], dataset_cfg["dataset_config"], split="train"
    )
    dataset["valid"] = JetSubstructureDataset(
        dataset_cfg["dataset_file"], dataset_cfg["dataset_config"], split="train"
    )  # This dataset is so small, we'll just use the training set as the validation set, otherwise we may have too few trainings examples to converge.
    dataset["test"] = JetSubstructureDataset(
        dataset_cfg["dataset_file"], dataset_cfg["dataset_config"], split="test"
    )
    
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="PolyLUT_Add",
        # track hyperparameters and run metadata
        config={
            "hidden_layers": model_cfg["hidden_layers"],
            "input_bitwidth": model_cfg["input_bitwidth"],
            "hidden_bitwidth": model_cfg["hidden_bitwidth"],
            "output_bitwidth": model_cfg["output_bitwidth"],
            "input_fanin": model_cfg["input_fanin"],
            "degree": model_cfg["degree"],
            "hidden_fanin": model_cfg["hidden_fanin"],
            "output_fanin": model_cfg["output_fanin"],
            "weight_decay": train_cfg["weight_decay"],
            "batch_size": train_cfg["batch_size"],
            "epochs": train_cfg["epochs"],
            "learning_rate": train_cfg["learning_rate"],
            "seed": train_cfg["seed"],
            "dataset": "jsc",
        },
    )
    

    # Instantiate model
    fre = 2.4
    #while(fre <= 10):
    x, y = dataset["train"][0]
    model_cfg["input_length"] = len(x)
    model_cfg["output_length"] = len(y)
    
    
    # select PolyLUT model or PolyLUT-Add model
    model = JetSubstructureNeqModel_add2(model_cfg)

    
    if options_cfg["checkpoint"] is not None:
        print(f"Loading pre-trained checkpoint {options_cfg['checkpoint']}")
        checkpoint = torch.load(options_cfg["checkpoint"], map_location="cpu")
        model.load_state_dict(checkpoint["model_dict"])
    
    
    wandb.define_metric("Train Acc (%)", summary="max")
    wandb.define_metric("Test Acc (%)", summary="max")
    wandb.define_metric("Valid Acc(%)", summary="max")
    wandb.define_metric("Train Loss(%)", summary="min")
    wandb.watch(model, log_freq=10)
    
    train(model, dataset, train_cfg, options_cfg, fre)
        #fre += 0.1
    wandb.finish()
