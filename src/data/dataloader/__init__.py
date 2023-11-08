from .ade import ADE20KSegmentation

datasets = {
    'ADE20K': ADE20KSegmentation,
    # 'pascal_voc': VOCSegmentation,
    # 'pascal_aug': VOCAugSegmentation,
    # 'coco': COCOSegmentation,
    # 'citys': CitySegmentation,
    # 'sbu': SBUSegmentation,
}

def get_segmentation_dataset(name, **kwargs):
    """Segmentation Datasets"""
    return datasets[name.lower()](**kwargs)