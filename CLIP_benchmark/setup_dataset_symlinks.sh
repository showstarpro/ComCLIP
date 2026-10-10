#!/bin/bash
# Symlink the clip-benchmark datasets under /path/to/data into the directory layout expected by clip_benchmark
# Expected layout: CLIP_benchmark/datasets/wds_wds-{name}/  (containing .tar files)
# Source layout:   /path/to/data/clip-benchmark__wds_{name}/{branch}/

set -e

DEST_DIR="$(dirname "$0")/datasets"
SRC_BASE="/path/to/data"

mkdir -p "$DEST_DIR"

# Mapping: dataset_cleaned -> source dataset directory name
# Format: "cleaned_name:source_dir_name:branch"
DATASETS=(
    "vtab-cifar10:clip-benchmark__wds_vtab-cifar10:main"
    "vtab-cifar100:clip-benchmark__wds_vtab-cifar100:main"
    "vtab-caltech101:clip-benchmark__wds_vtab-caltech101:main"
    "fer2013:clip-benchmark__wds_fer2013:23-01-31-0026"
    "vtab-pets:clip-benchmark__wds_vtab-pets:main"
    "vtab-dtd:clip-benchmark__wds_vtab-dtd:23-01-20-0712"
    "vtab-resisc45:clip-benchmark__wds_vtab-resisc45:23-01-20-0722"
    "vtab-eurosat:clip-benchmark__wds_vtab-eurosat:23-01-20-0713"
    "vtab-pcam:clip-benchmark__wds_vtab-pcam:23-01-20-0720"
    "imagenet_sketch:clip-benchmark__wds_imagenet_sketch:23-01-20-0558"
    "imagenet-o:clip-benchmark__wds_imagenet-o:main"
)

echo "Creating symlinks in: $DEST_DIR"
echo ""

for entry in "${DATASETS[@]}"; do
    IFS=':' read -r cleaned src_dir branch <<< "$entry"
    link_name="wds_${cleaned}"
    src_path="${SRC_BASE}/${src_dir}/${branch}"

    if [ ! -d "$src_path" ]; then
        echo "WARNING: Source not found: $src_path (skipping)"
        continue
    fi

    target="${DEST_DIR}/${link_name}"
    if [ -L "$target" ]; then
        echo "EXISTS:  $link_name -> $(readlink "$target")"
    elif [ -d "$target" ]; then
        echo "EXISTS (dir): $link_name (not a symlink, skipping)"
    else
        ln -s "$src_path" "$target"
        echo "LINKED:  $link_name -> $src_path"
    fi
done

echo ""
echo "Done. Current datasets directory:"
ls -la "$DEST_DIR"
