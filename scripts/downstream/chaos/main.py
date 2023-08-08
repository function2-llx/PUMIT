from luolib.utils.lightning import LightningCLI

class CLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        super().add_arguments_to_parser(parser)
        parser.link_arguments('trainer.max_steps', 'data.init_args.num_train_batches')

def main():
    import torch.multiprocessing
    torch.multiprocessing.set_sharing_strategy('file_system')
    CLI()

if __name__ == '__main__':
    main()
