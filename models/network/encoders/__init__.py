import functools
import torch.utils.model_zoo as model_zoo

from .resnet import instantiate_encoder


from .timm_universal import TimmUniversalEncoder

from ._preprocessing import preprocess_input

import torch

def get_encoder(name, data_path, in_channels=3, depth=5, weights=None, output_stride=32, **kwargs):
    
    encoders = instantiate_encoder(data_path)

    if name.startswith("tu-"):
        name = name[3:]
        encoder = TimmUniversalEncoder(
            name=name,
            in_channels=in_channels,
            depth=depth,
            output_stride=output_stride,
            pretrained=weights is not None,
            **kwargs
        )
        return encoder

    try:
        Encoder = encoders[name]["encoder"]
    except KeyError:
        raise KeyError("Wrong encoder name `{}`, supported encoders: {}".format(name, list(encoders.keys())))

    params = encoders[name]["params"]
    params.update(depth=depth)
    encoder = Encoder(**params)

    if weights is not None:
        try:
            settings = encoders[name]["pretrained_settings"][weights]
        except KeyError:
            raise KeyError("Wrong pretrained weights `{}` for encoder `{}`. Available options are: {}".format(
                weights, name, list(encoders[name]["pretrained_settings"].keys()),
            ))
        pretrained_weights = torch.load(settings["path"])
        encoder.load_state_dict(pretrained_weights, strict=True)
        print(f"loaded weights from {settings['path']} into the resnet encoder!")

    encoder.set_in_channels(in_channels, pretrained=weights is not None)
    if output_stride != 32:
        encoder.make_dilated(output_stride)
    
    return encoder