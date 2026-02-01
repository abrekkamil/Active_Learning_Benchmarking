def load_dataset(config, split):
    if config.dataset_type == "deepcrack":
        from .deepcrack import DeepCrackSegmentationDataset
        return DeepCrackSegmentationDataset(
            config.data_dir, split, config.img_size
        )

    if config.dataset_type == "coco_detection":
        from .coco_detection import CocoDetectionDataset
        return CocoDetectionDataset(
            config.data_dir, split, is_train=(split == "train")
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

    raise ValueError(f"Unknown dataset type: {config.dataset_type}")
