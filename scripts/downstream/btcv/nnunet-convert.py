import json
from pathlib import Path

from nnunetv2.paths import nnUNet_raw, nnUNet_preprocessed
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json

def extract_keys(cases: list[dict]) -> list[str]:
    ret = []
    for case in cases:
        case_id = Path(case['image']).name.split('.')[0][-4:]
        ret.append(f'BTCV_{case_id}')
    return ret

def force_symlink_to(src: Path, dst: Path):
    if src.exists():
        src.unlink()
    src.symlink_to(dst)

def main():
    dataset_name = 'BTCV'
    raw_path = Path(nnUNet_raw) / f'Dataset666_{dataset_name}'
    preprocessed_path = Path(nnUNet_preprocessed) / f'Dataset666_{dataset_name}'
    raw_path.mkdir(exist_ok=True, parents=True)
    data_dir = Path('downstream/data/BTCV/Training')
    images_dir = raw_path / 'imagesTr'
    labels_dir = raw_path / 'labelsTr'
    images_dir.mkdir(exist_ok=True)
    labels_dir.mkdir(exist_ok=True)
    for img_path in (data_dir / 'img').iterdir():
        case_id = img_path.name.split('.')[0][-4:]
        force_symlink_to(images_dir / f'BTCV_{case_id}_0000.nii.gz', img_path.resolve())
        force_symlink_to(labels_dir / f'BTCV_{case_id}.nii.gz', (data_dir / 'label' / f'label{case_id}.nii.gz').resolve())

    generate_dataset_json(
        str(raw_path),
        {'0': 'CT'},
        {
            'background': '0',
            'spleen': '1',
            'rkid': '2',
            'lkid': '3',
            'gall': '4',
            'eso': '5',
            'liver': '6',
            'sto': '7',
            'aorta': '8',
            'IVC': '9',
            'veins': '10',
            'pancreas': '11',
            'rad': '12',
            'lad': '13',
        },
        30,
        '.nii.gz',
        dataset_name=dataset_name,
    )
    split = json.loads(Path('downstream/data/BTCV/smit.json').read_bytes())

    (preprocessed_path / 'splits_final.json').write_text(
        json.dumps(
            [
                {
                    'train': extract_keys(split['training']),
                    'val': extract_keys(split['validation']),
                }
            ],
            indent=4,
            ensure_ascii=False,
        ),
    )

if __name__ == '__main__':
    main()
