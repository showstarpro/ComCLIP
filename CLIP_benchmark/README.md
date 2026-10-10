Dataset root directory layout
--dataset_root takes a template: ../CLIP_benchmark/datasets/wds_{dataset_cleaned}

For each dataset in datasets.txt, the expanded path is:

Name in datasets.txt	{dataset_cleaned}	Actual path
wds/vtab/cifar10	wds-vtab-cifar10	CLIP_benchmark/datasets/wds_wds-vtab-cifar10/
wds/vtab/cifar100	wds-vtab-cifar100	CLIP_benchmark/datasets/wds_wds-vtab-cifar100/
wds/vtab/caltech101	wds-vtab-caltech101	CLIP_benchmark/datasets/wds_wds-vtab-caltech101/
wds/fer2013	wds-fer2013	CLIP_benchmark/datasets/wds_wds-fer2013/
wds/vtab/pets	wds-vtab-pets	CLIP_benchmark/datasets/wds_wds-vtab-pets/
wds/vtab/dtd	wds-vtab-dtd	CLIP_benchmark/datasets/wds_wds-vtab-dtd/
wds/vtab/resisc45	wds-vtab-resisc45	CLIP_benchmark/datasets/wds_wds-vtab-resisc45/
wds/vtab/eurosat	wds-vtab-eurosat	CLIP_benchmark/datasets/wds_wds-vtab-eurosat/
wds/vtab/pcam	wds-vtab-pcam	CLIP_benchmark/datasets/wds_wds-vtab-pcam/
wds/imagenet_sketch	wds-imagenet_sketch	CLIP_benchmark/datasets/wds_wds-imagenet_sketch/
wds/imagenet-o	wds-imagenet-o	CLIP_benchmark/datasets/wds_wds-imagenet-o/
So the datasets/ directory should look like this:

CLIP_benchmark/datasets/
├── wds_wds-vtab-cifar10/
│   ├── 00000.tar
│   ├── 00001.tar
│   └── ...
├── wds_wds-vtab-cifar100/
│   ├── 00000.tar
│   └── ...
├── wds_wds-vtab-caltech101/
├── wds_wds-fer2013/
├── wds_wds-vtab-pets/
├── wds_wds-vtab-dtd/
├── wds_wds-vtab-resisc45/
├── wds_wds-vtab-eurosat/
├── wds_wds-vtab-pcam/
├── wds_wds-imagenet_sketch/
└── wds_wds-imagenet-o/
Each subdirectory holds .tar shards in WebDataset (wds) format, the standard data format used by clip_benchmark.