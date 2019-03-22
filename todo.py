# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import subprocess
from shlex import split
import os.path as osp
import sys


scales = [500, 800]

commands = []

# Zoo

for s in scales:
    c = (f"scripts/finetune_p.py --expdir experiments/zoo_1B_1.0/finetune{s} --init-p 1.0"
         f" --pretrained-backbone"
         f" --imagenet-path /esat/visicsrodata/datasets/ilsvrc2012 --input-size {s}"
         f" --resume-epoch 0 --images-per-class 50")
    commands.append((c, f"experiments/zoo_1B_1.0/finetune{s}"))

# Resnet-50

for s in scales:
    for p in [1, 3]:
        c = (f"scripts/finetune_p.py --expdir experiments/base_{p}B_1.0/finetune{s}"
             f" --init-p {p} --imagenet-path /esat/visicsrodata/datasets/ilsvrc2012"
             f" --input-size {s}"
             f" --resume-epoch -1 --images-per-class 50 --resume-from experiments/base_{p}B_1.0")
        commands.append((c, f"experiments/base_{p}B_1.0/finetune{s}"))

# MultiGrain

for s in scales:
    for p in [1, 3]:
        for w in [0.5, 1.0]:
            c = (f"scripts/finetune_p.py --expdir experiments/joint_{p}B_{w}/finetune{s}"
                 f" --init-p {p} --imagenet-path /esat/visicsrodata/datasets/ilsvrc2012"
                 f" --input-size {s}"
                 f" --resume-epoch -1 --images-per-class 50 --resume-from experiments/joint_{p}B_{w}")
            commands.append((c, f"experiments/joint_{p}B_{w}/finetune{s}"))

# AA
for s in scales:
    for p in [1, 3]:
        c = (f"scripts/finetune_p.py --expdir experiments/joint_{p}BAA+_0.5/finetune{s}"
             f" --init-p {p} --imagenet-path /esat/visicsrodata/datasets/ilsvrc2012"
             f" --input-size {s} "
             f" --resume-epoch -1 --images-per-class 50 --resume-from experiments/joint_{p}BAA+_0.5")
        commands.append((c, f"experiments/joint_{p}BAA+_0.5/finetune{s}"))

# print(commands)

running = [(i.split()[0], i.split()[1:]) for i in subprocess.check_output(split('condor_q -af ClusterID Arguments')).decode().split('\n') if i]
# print(running)
# running = [(int(i[0]), i[1:]) for i in running]

#print(running)

for i, e in enumerate(commands):
    if osp.isfile(osp.join(e[1], 'block')):
        print("blocked:", e[1])
        continue
    for r in running:
        # print("'" + e[1] + "'", r[1])
        if "'" + e[1] + "'" in r[1]:
            print('found', e[1], ":", r[0])
            break
    else:
        if 'dry' in sys.argv:
            print('python', e[0])
            continue
        # print(e[0])
        # # print("not found", e[1])
        comm = (f"condor_send --gpus 1 --gpumem 6 --mem 8 --cpus 10 --timeout .1 --jobname finetune --machineowner Visics "
                f"--type conda --conda-env multigrain -c '{e[0]}'")
        # # print(comm)
        os.system(comm)
