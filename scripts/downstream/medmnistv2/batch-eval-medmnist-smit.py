import os

databases = [
#    "adrenalmnist3d",
#    "fracturemnist3d",
    "nodulemnist3d",
    "organmnist3d",
    "synapsemnist3d",
#    "vesselmnist3d",
]

c = [1, 2]

lr = 5e-7
interpolate = True

for (i, db) in enumerate(databases):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
    os.system(
        f"python scripts/evaluation/eval-medmnist.py --model_path third-party/SMIT/Pre_trained/pre_train_weight.pt \
        --arch SMIT --data_flag {db} --run lr_{lr:.0e} --lr {lr}"
        + (
        " --interpolate"
        if interpolate
        else ""
        ) + (
        " --shape_transform"
        if db in ["adrenalmnist3d", "vesselmnist3d"]
        else ""
        ) + f" > smit_{db}_lr_{lr:.0e}{'_interpolate' if interpolate else ''}.log 2>&1"
    )