from torch.nn import functional as F

from torch import nn, distributions

# Detectron imports
import fvcore.nn.weight_init as weight_init

from detectron2.config import configurable
from detectron2.layers import Linear, ShapeSpec, cat, Conv2d, get_norm
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN
from detectron2.modeling.roi_heads import ROI_HEADS_REGISTRY, StandardROIHeads
from detectron2.modeling.roi_heads.fast_rcnn import fast_rcnn_inference
from detectron2.modeling.roi_heads.box_head import ROI_BOX_HEAD_REGISTRY
from detectron2.structures import Boxes, Instances, ImageList
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import log_first_n

import torch
from typing import Dict, List, Optional, Tuple

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu")

@META_ARCH_REGISTRY.register()
class EvidentialRCNN(GeneralizedRCNN):
    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]],epoch=1, total_epoch=10):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
            epoch: current epoch
            total_epoch : total epochs

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        if not self.training:
            return self.inference(batched_inputs)

        self.epoch = epoch
        self.total_epoch = total_epoch

        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)

        if self.proposal_generator is not None:
            proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        _, detector_losses = self.roi_heads(images, features, proposals, gt_instances,epoch=epoch,total_epoch=total_epoch)
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        losses = {}
        losses.update(detector_losses)
        # losses.update(proposal_losses)
        return losses