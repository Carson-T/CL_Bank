import timm
from safetensors.torch import load_file
from timm.models import register_model


@register_model
def MyMobileNet(backbone, pretrained_path, is_pretrained=False, **kwargs):
    backbone = timm.create_model(backbone, drop_rate=kwargs["drop_rate"], drop_path_rate=kwargs["drop_path_rate"], pretrained=is_pretrained)
    if pretrained_path:
        backbone.load_state_dict(load_file(pretrained_path))
    feature_dim = backbone.classifier.in_features

    # return backbone, feature_dim
    return backbone