__all__ = [
    'is_natural_modality',
]

def is_natural_modality(modality: str) -> bool:
    return modality.startswith('RGB') or modality.startswith('gray')
