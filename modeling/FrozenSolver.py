from detectron2.solver.build import get_default_optimizer_params, CfgNode, TORCH_VERSION, maybe_add_gradient_clipping
import torch


def build_optimizer_backbone_rpn_frozen(cfg: CfgNode, model: torch.nn.Module) -> torch.optim.Optimizer:
    """
    Build an optimizer from config.
    """
    for name, para in model.named_parameters():
        if 'roi_head' in name or 'box_predictor' in name:
            para.requires_grad = True  # print(name)
        else: para.requires_grad = False

    params = get_default_optimizer_params(
        model,
        base_lr=cfg.SOLVER.BASE_LR,
        weight_decay_norm=cfg.SOLVER.WEIGHT_DECAY_NORM,
        bias_lr_factor=cfg.SOLVER.BIAS_LR_FACTOR,
        weight_decay_bias=cfg.SOLVER.WEIGHT_DECAY_BIAS,
    )

    sgd_args = {
        "params": params,
        "lr": cfg.SOLVER.BASE_LR,
        "momentum": cfg.SOLVER.MOMENTUM,
        "nesterov": cfg.SOLVER.NESTEROV,
        "weight_decay": cfg.SOLVER.WEIGHT_DECAY,
    }
    if TORCH_VERSION >= (1, 12):
        sgd_args["foreach"] = True
    return maybe_add_gradient_clipping(cfg, torch.optim.SGD(**sgd_args))

