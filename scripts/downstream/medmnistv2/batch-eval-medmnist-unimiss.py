import os

databases = [
    "adrenalmnist3d",
    "fracturemnist3d",
    "nodulemnist3d",
    "organmnist3d",
    "synapsemnist3d",
    "vesselmnist3d",
]

c = [0, 1, 2]

lr = 1e-7
interpolate = True

for (i, db) in enumerate(databases):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(1)
    os.system(
        f"python scripts/evaluation/eval-medmnist.py --model_path ../UniMiSS-code/UniMiss_small.pth \
        --arch UniMiSS_small --data_flag {db} --run lr_{lr:.0e} --lr {lr} --interpolate"
        + (
        " --interpolate"
        if interpolate
        else ""
        ) + (
        " --shape_transform"
        if db in ["adrenalmnist3d", "vesselmnist3d"]
        else ""
        ) + f" > model_small_{db}_2_lr_{lr:.0e}{'_interpolate' if interpolate else ''}.log 2>&1"
    )