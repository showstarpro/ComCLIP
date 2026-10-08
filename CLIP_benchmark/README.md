数据集根目录结构
--dataset_root 传入的是模板：../CLIP_benchmark/datasets/wds_{dataset_cleaned}

对于 datasets.txt 中的每个数据集，实际展开后的路径如下：

datasets.txt 中的名称	{dataset_cleaned}	实际路径
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
所以 datasets/ 目录应该长这样

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
每个子目录里存放的是 WebDataset (wds) 格式的 .tar 分片文件。这是 clip_benchmark 使用的标准数据格式。