_base_ = [
    '../_base_/models/fcn_r50-d8.py', '../_base_/datasets/nightcity.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]
data=dict(samples_per_gpu=8, workers_per_gpu=8)