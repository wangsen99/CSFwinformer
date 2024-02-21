_base_ = [
    '../_base_/models/fcn_r50-d8.py', '../_base_/datasets/nightcity.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]
model = dict(
    pretrained='open-mmlab://resnet18_v1c',
    backbone=dict(depth=18),
    decode_head=dict(
        in_channels=512,
        channels=128,
    ),
    auxiliary_head=dict(in_channels=256, channels=64))
data=dict(samples_per_gpu=8, workers_per_gpu=8)