import fiftyone as fo
import fiftyone.zoo as foz

# The path to the source files that you manually downloaded
source_dir = "E:\Datasets\Classification"

dataset = foz.load_zoo_dataset(
    "imagenet-2012",
    split="validation",
    source_dir=source_dir,
)

session = fo.launch_app(dataset,port=5150)
session.wait()