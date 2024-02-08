from operator import xor

from torch.utils.data import ConcatDataset, DataLoader

import dataset as ds_module
from utils.parse_config import ConfigParser
from augmentations import get_augmentations


def get_dataloaders(configs: ConfigParser):
    dataloaders = {}
    for split, params in configs["data"].items():
        num_workers = params.get("num_workers", 1)

        # set train augmentations
        augs = get_augmentations((split == 'train'))

        # create and join datasets
        datasets = []
        for ds in params["datasets"]:
            datasets.append(configs.init_obj(
                ds, ds_module, config_parser=configs,
                transforms = augs))
        assert len(datasets)
        if len(datasets) > 1:
            dataset = ConcatDataset(datasets)
        else:
            dataset = datasets[0]

        # select batch size or batch sampler
        assert xor("batch_size" in params, "batch_sampler" in params), \
            "You must provide batch_size or batch_sampler for each split"
        if "batch_size" in params:
            bs = params["batch_size"]
            shuffle = True
            batch_sampler = None
        else:
            raise Exception()

        # Fun fact. An hour of debugging was wasted to write this line
        assert bs <= len(dataset), \
            f"Batch size ({bs}) shouldn't be larger than dataset length ({len(dataset)})"

        # create dataloader
        dataloader = DataLoader(
            dataset, batch_size=bs, collate_fn=ds_module.collate_fn,
            shuffle=shuffle, num_workers=num_workers,
            batch_sampler=batch_sampler
        )
        dataloaders[split] = dataloader
    return dataloaders