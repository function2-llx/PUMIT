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

lr = 5e-8
interpolate = True

for (i, db) in enumerate(databases):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(c[(i + 1) % 3])
    os.system(
        f"python scripts/evaluation/eval-medmnist.py --model_path ../eva02_B_pt_in21k_p14.pt \
        --arch EVA_02_B --data_flag {db} --run lr_{lr:.0e} --lr {lr}"
        + (
        " --interpolate"
        if interpolate
        else ""
        ) + (
        " --shape_transform"
        if db in ["adrenalmnist3d", "vesselmnist3d"]
        else ""
        ) + f" > eva02_b_{db}_lr_{lr:.0e}{'_interpolate' if interpolate else ''}.log 2>&1 &"
    )