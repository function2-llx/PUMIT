import os

datasets = [
    "adrenalmnist3d",
    "fracturemnist3d",
    "nodulemnist3d",
    "organmnist3d",
    "synapsemnist3d",
    "vesselmnist3d",
    "retinamnist",
    "dermamnist",
    "tissuemnist",
]

lrs = [
    5e-7,
    1e-6,
    5e-6,
    1e-5,
    5e-5,
    1e-4,
    5e-4
]

interpolate = True

for lr in lrs:
    for (i, db) in enumerate(datasets):
        print(f"{db} {lr}")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
        os.system(
            f"python downstream/medmnistv2/eval-medmnist.py --model_path /data/PUMIT/archive/pumit-b.ckpt \
            --arch PUMIT --data_flag {db} --run lr_{lr:.0e} --lr {lr}"
            + (
            " --interpolate"
            if interpolate
            else ""
            ) + (
            " --shape_transform"
            if db in ["adrenalmnist3d", "vesselmnist3d"]
            else ""
            ) + f" > pumit_{db}_lr_{lr:.0e}{'_interpolate' if interpolate else ''}.log 2>&1"
        )