import torch
from lightning.pytorch.cli import LightningCLI

def main():
    torch.set_float32_matmul_precision('high')
    LightningCLI(
        save_config_kwargs={'overwrite': True},
    )

if __name__ == "__main__":
    main()
