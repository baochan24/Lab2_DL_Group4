from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torchvision import models

_SUPPORTED_MODELS = {"vgg16", "densenet121"}
_SUPPORTED_STRATEGIES = {"classifier_only", "finetune_last_layers"}

def _get_pretrained_weights(model_name: str, pretrained: bool):
    """Return the appropriate Torchvision weights enum or None."""
    if not pretrained:
        return None
    if model_name == "vgg16":
        return models.VGG16_Weights.DEFAULT
    if model_name == "densenet121":
        return models.DenseNet121_Weights.DEFAULT
    raise ValueError(f"Unsupported model_name: {model_name!r}")

def _replace_classifier(model: nn.Module, model_name: str, num_classes: int) -> None:
    """Replace the model's final classification layer in-place."""
    if model_name == "vgg16":
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    elif model_name == "densenet121":
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
    else:  # defensive; model_name được xác thực trong build_model
        raise ValueError(f"Unsupported model_name: {model_name!r}")

def _configure_trainable_parameters(
    model: nn.Module,
    model_name: str,
    strategy: str,
    finetune_layers: list[str] | tuple[str, ...] | None,
) -> list[str]:
    """Freeze the backbone and enable the classifier and requested layers.

    Returns the names of explicitly selected fine-tuning prefixes. Prefixes are
    validated against actual parameter names so a typo cannot silently freeze
    the entire backbone.
    """
    for parameter in model.parameters():
        parameter.requires_grad = False

    # Bộ phân loại luôn có thể huấn luyện được cho cả hai chiến lược được hỗ trợ.
    classifier = model.classifier
    for parameter in classifier.parameters():
        parameter.requires_grad = True

    if strategy == "classifier_only":
        return []

    if not finetune_layers:
        raise ValueError(
            "strategy='finetune_last_layers' requires a non-empty "
            "config['finetune_layers'] list of module-name prefixes."
        )

    named_parameters = list(model.named_parameters())
    valid_names = [name for name, _ in named_parameters]
    unmatched = [
        prefix for prefix in finetune_layers
        if not any(name == prefix or name.startswith(prefix + ".") for name in valid_names)
    ]
    if unmatched:
        examples = ", ".join(valid_names[:12])
        raise ValueError(
            f"Unknown finetune_layers prefix(es): {unmatched}. "
            f"Check names for {model_name}; sample parameter names: {examples}"
        )

    #Giữ cho bộ phân loại có thể huấn luyện được và chọn lọc bỏ đóng băng các phần xương sống được yêu cầu.
    for name, parameter in named_parameters:
        if any(name == prefix or name.startswith(prefix + ".") for prefix in finetune_layers):
            parameter.requires_grad = True

    return list(finetune_layers)

def get_model_summary(model: nn.Module) -> dict[str, int]:
    """Return total and trainable parameter counts."""
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return {"total_parameters": total, "trainable_parameters": trainable}

def set_frozen_batchnorm_eval(model: nn.Module) -> None:
    """Keep BatchNorm modules with no trainable parameters in eval mode.

    Call this immediately after ``model.train()`` in every training epoch.
    ``requires_grad=False`` freezes affine parameters but does not freeze
    BatchNorm running_mean/running_var buffers by itself. BatchNorm modules
    whose parameters are intentionally unfrozen for fine-tuning remain in train mode.
    """
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            parameters = list(module.parameters(recurse=False))
            if parameters and not any(parameter.requires_grad for parameter in parameters):
                module.eval()

def build_model(config: dict[str, Any]) -> nn.Module:
    """Build a pretrained VGG16 or DenseNet121 configured for CIFAR-10.

    Args:
        config: Dictionary loaded from the experiment YAML.

    Returns:
        A Torchvision model that outputs raw logits with shape ``[B, num_classes]``.

    Raises:
        ValueError: If model name, strategy, class count, or fine-tune layers are invalid.
    """
    if not isinstance(config, dict):
        raise TypeError("config must be a dictionary")

    model_name = str(config.get("model_name", "")).strip().lower()
    if model_name not in _SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported model_name={model_name!r}. Choose one of {sorted(_SUPPORTED_MODELS)}."
        )

    pretrained = config.get("pretrained", True)
    if not isinstance(pretrained, bool):
        raise TypeError("config['pretrained'] must be a bool")

    num_classes = config.get("num_classes", 10)
    if not isinstance(num_classes, int) or isinstance(num_classes, bool) or num_classes < 2:
        raise ValueError("config['num_classes'] must be an integer >= 2")

    strategy = str(config.get("strategy", "classifier_only")).strip().lower()
    if strategy not in _SUPPORTED_STRATEGIES:
        raise ValueError(
            f"Unsupported strategy={strategy!r}. Choose one of {sorted(_SUPPORTED_STRATEGIES)}."
        )

    weights = _get_pretrained_weights(model_name, pretrained)
    if model_name == "vgg16":
        model = models.vgg16(weights=weights)
    else:
        model = models.densenet121(weights=weights)

    _replace_classifier(model, model_name, num_classes)
    selected_layers = _configure_trainable_parameters(
        model=model,
        model_name=model_name,
        strategy=strategy,
        finetune_layers=config.get("finetune_layers"),
    )

    summary = get_model_summary(model)
    model.tv2_metadata = {
        "model_name": model_name,
        "pretrained": pretrained,
        "weights": f"{model_name.upper()}_Weights.DEFAULT" if pretrained else None,
        "num_classes": num_classes,
        "strategy": strategy,
        "finetune_layers": selected_layers,
        **summary,
    }

    print("=" * 60)
    print("TV2 model summary")
    print(f"Architecture       : {model_name}")
    print(f"Pretrained weights : {model.tv2_metadata['weights'] or 'None (random init)'}")
    print(f"Classes            : {num_classes}")
    print(f"Strategy           : {strategy}")
    print(f"Fine-tune prefixes : {selected_layers or 'None'}")
    print(f"Total parameters   : {summary['total_parameters']:,}")
    print(f"Trainable params   : {summary['trainable_parameters']:,}")
    print("Output             : raw logits (no Softmax)")
    print("=" * 60)

    return model

__all__ = ["build_model", "get_model_summary", "set_frozen_batchnorm_eval"]