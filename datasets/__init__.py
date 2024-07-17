import torch.utils.data
import torchvision

from .ytvos import build as build_ytvos
from .davis import build as build_davis
from .a2d import build as build_a2d
from .jhmdb import build as build_jhmdb
from .refexp import build as build_refexp
from .concat_dataset import build as build_joint
from .mot17_in_ytvos import build as build_mot17
from .ovis_in_ytvos import build as build_ovis


def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco


def build_dataset(dataset_file: str, image_set: str, args):
    if dataset_file == 'ytvos':
        if args.mot17:
            return build_mot17(image_set, args)
        elif args.ovis:
            return build_ovis(image_set, args)
        else:
            return build_ytvos(image_set, args)
    if dataset_file == 'davis':
        return build_davis(image_set, args)
    if dataset_file == 'a2d':
        return build_a2d(image_set, args)
    if dataset_file == 'jhmdb':
        return build_jhmdb(image_set, args)
    # for pretraining
    if dataset_file == "refcoco" or dataset_file == "refcoco+" or dataset_file == "refcocog":
        return build_refexp(dataset_file, image_set, args)
    # for joint training of refcoco and ytvos
    if dataset_file == 'joint':
        return build_joint(image_set, args)
    raise ValueError(f'dataset {dataset_file} not supported')
