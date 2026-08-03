def load_dataset(config, split):
    # Leakage-safe three-way split controls.
    #   val_source: "train" -> val carved from train/, test = full test/
    #               "test"  -> train kept whole, val+test are seeded halves of test/
    #   val_fraction: size of the val slice (of train, or of test)
    # Defaults reproduce old behaviour only if val_fraction is set to 0.0.
    val_source = getattr(config, "val_source", "train")
    val_fraction = getattr(config, "val_fraction", 0.2)
    val_seed = getattr(config, "val_seed", 12345)

    if config.dataset_type == "deepcrack":
        from .deepcrack import DeepCrackSegmentationDataset
        return DeepCrackSegmentationDataset(
            config.data_dir, split, config.img_size,
            val_source=val_source, val_fraction=val_fraction, val_seed=val_seed,
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
            config.data_dir, split, img_size=config.img_size,
            val_source=val_source, val_fraction=val_fraction, val_seed=val_seed,
        )
    if config.dataset_type == "sewerml":
        print(f"Loading Sewerml dataset with split: {split}")
        from .sewerml import MultiLabelDataset
        if split == 'val':
            split = 'valid' # Sewerml only has Train and Test splits, so we use Test for validation
        return MultiLabelDataset(
            args=config,
            img_dir=config.data_dir,
            labels_path=config.data_dir,
            testing=False,
            split= split
        )
    if config.dataset_type == "yolo_segmentation":
        from .yolo_instance_segmentation import YoloInstanceSegmentationDataset
        return YoloInstanceSegmentationDataset(
            config.data_dir, split, config.img_size
        )

    raise ValueError(f"Unknown dataset type: {config.dataset_type}")