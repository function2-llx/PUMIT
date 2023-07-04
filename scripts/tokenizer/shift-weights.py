from pathlib import Path

import torch

def main():
    ckpt_path = Path('pre-trained/tokenizer/vq-oi-f8-8192-gq/model.ckpt')
    save_path = Path('pre-trained/tokenizer/vq-oi-f8-8192-gq/model-shifted.ckpt')
    ckpt = torch.load(ckpt_path)
    state_dict: dict[str, torch.Tensor] = ckpt['state_dict']
    shifted = {}
    for key, weight in state_dict.items():
        for prefix in ['encoder.down.', 'decoder.up.']:
            if key.startswith(prefix):
                i, suffix = key[len(prefix):].split('.', 1)
                print(key, i, suffix)
                shifted[f'{prefix}{int(i) + 1}.{suffix}'] = weight
                break
        else:
            shifted[key] = weight
    ckpt['state_dict'] = shifted
    torch.save(ckpt, save_path)

if __name__ == "__main__":
    main()
