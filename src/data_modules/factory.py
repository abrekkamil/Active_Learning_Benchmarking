def load_dataset(config, split):
    if config.dataset_type == "deepcrack":
        from .deepcrack import DeepCrackSegmentationDataset
        return DeepCrackSegmentationDataset(
            config.data_dir, split, config.img_size
        )

    if config.dataset_type == "coco_instance":
        from .coco_instance import CocoInstanceSegmentationDataset
        return CocoInstanceSegmentationDataset(
            config.data_dir, split, is_train=True
        )

    if config.dataset_type == "coco_segmentation":
        from .coco_segmentation import CocoSemanticSegmentationDataset
        return CocoSemanticSegmentationDataset(
            config.data_dir, split, config.img_size
        )
    if config.dataset_type == "crackseg9k":
        from .CrackSeg9K import CrackSeg9KDataset
        return CrackSeg9KDataset(
            config.data_dir, split, img_size=config.img_size
        )
    if config.dataset_type == "sewerml":
        from .sewerml import MultiLabelDataset
        return MultiLabelDataset(
            img_dir=config.data_dir,
            labels_path=config.data_dir,
            testing=False,
            split= split
        )
    raise ValueError(f"Unknown dataset type: {config.dataset_type}")
