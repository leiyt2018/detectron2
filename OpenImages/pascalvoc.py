import fiftyone as fo
import fiftyone.zoo as foz

dataset = foz.load_zoo_dataset("voc-2012", split="validation")

session = fo.launch_app(dataset,port=5156)
session.wait()