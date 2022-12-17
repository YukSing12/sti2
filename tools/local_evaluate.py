#!/usr/bin/env python
# encoding=utf8
#ras-sat ernie模型
#评价方式：Metric损失率=abs(原始Metric-优化后Metric)/原始Metric

#1个输入：优化后模型打分文件
#       打分文件命名格式：参赛label.res.txt和perf.res.txt
#       打分文件每行格式：query_id、label_id、score_list、微秒时间戳共4个输出，字符串格式且以\t分割
#                         多batch的score_list以逗号分割

#2个输出：metric: 数值越大越好
#    推理平均时延：数值越小越好

import sys
import os
import math


def metric(qid, label, pred):
    """
    input:
        qid: query id(int)
        label: label id(int/float)
        pred: score(float)
    output:
        m: float
    """
    saver = {}
    assert len(qid) == len(label) == len(pred)
    for q, l, p in zip(qid, label, pred):
        if q not in saver:
            saver[q] = []
        saver[q].append((l, p))
    p = 0
    n = 0
    for qid, outputs in saver.items():
        for i in range(0, len(outputs)):
            l1, p1 = outputs[i]
            for j in range(i + 1, len(outputs)):
                l2, p2 = outputs[j]
                if l1 > l2:
                    if p1 > p2:
                        p += 1
                    elif p1 < p2:
                        n += 1
                elif l1 < l2:
                    if p1 < p2:
                        p += 1
                    elif p1 > p2:
                        n += 1
    m = 1. * p / n if n > 0 else 0.0
    return m


def avg_inf(opt_tms):
    """
    input:
         opt_tms: [16666889979777.0,...]
    output:
         inference time: 0.12 us
    """
    opt_nums = len(opt_tms)
    opt_all_times = 0.0
    for i in range(1, opt_nums):
        opt_all_times += float(opt_tms[i]) - float(opt_tms[i-1])

    return 1.* (opt_all_times / (opt_nums-1))


def evalute(opt_list):
    """
    input:
        opt_list: ["qid\tlabel\tscore",...]
    output:
        result: {"metric":3.03, "inf_time(us)":0.12}
    """
    opt_qids = []
    opt_labels = []
    opt_scores = []
    opt_tms = []
    for line in opt_list:
        value = line.strip().split("\t")
        opt_qids.append(int(value[0]))
        if value[1] != "-":
            opt_labels.append(int(float(value[1])))
            opt_scores.append(float(value[2]))
        else:
            opt_scores.append(value[2])
        opt_tms.append(value[3])
    opt_metric = "-"
    if len(opt_labels):
        opt_metric = metric(opt_qids, opt_labels, opt_scores)
    result = {}
    result["metric"] = opt_metric
    result["inf_time(us)"] = avg_inf(opt_tms)

    return result


if __name__ == "__main__":
    """
    python local_evaluate.py label.res.txt
    """
    opt_list = []
    with open(sys.argv[1], 'r') as f:
        for line in f.readlines():
            opt_list.append(line.strip())

    print (evalute(opt_list))
