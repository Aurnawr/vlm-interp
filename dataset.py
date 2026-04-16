import os
from datasets import load_dataset, Image
from huggingface_hub import snapshot_download

# Download only text+image subsets and corresponding data folders
dataset_dir = snapshot_download(
    repo_id="ailor/AdvBench-omni",
    repo_type="dataset",
    local_dir="dataset",
    allow_patterns=[
        "advbench_t2i.jsonl",
        "benign_t2i.jsonl",
        "advbench_i.jsonl",
        "data/adv_seg_t2i/*",
        "data/benign_seg_images/*",
        "data/adv_i/*"
    ]
)

# Load the JSONL files locally and cast the image paths to Image objects 
# so PIL automatically loads them when accessing the dataset.
ds = load_dataset(
    "json",
    data_files=[
        os.path.join(dataset_dir, "advbench_t2i.jsonl"),
        os.path.join(dataset_dir, "benign_t2i.jsonl"),
        os.path.join(dataset_dir, "advbench_i.jsonl")
    ],
    split="train"
).cast_column("image", Image(decode=True))