import torch

from mylib.utils.lightning import LightningCLI

class CLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        super().add_arguments_to_parser(parser)
        parser.link_arguments('trainer.max_steps', 'data.init_args.num_train_batches')
        parser.link_arguments('data.init_args.sample_size', 'model.init_args.sample_size')

def main():
    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.set_float32_matmul_precision('high')
    CLI()

if __name__ == '__main__':
    main()
