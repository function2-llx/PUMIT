import torch

from mylib.utils.lightning import LightningCLI

class CLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments('seed_everything', 'data.init_args.seed')
        parser.link_arguments('trainer.max_steps', 'data.init_args.dl_conf.num_train_batches')

def main():
    torch.set_float32_matmul_precision('high')
    torch.multiprocessing.set_start_method('spawn', force=True)
    CLI()

if __name__ == "__main__":
    main()
