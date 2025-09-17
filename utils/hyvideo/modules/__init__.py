from .models import HYVideoDiffusionTransformer, HUNYUAN_VIDEO_CONFIG
from .models_inverse import HYVideoDiffusionTransformer as HYVideoDiffusionTransformer_inverse

def load_model(args, in_channels, out_channels, factor_kwargs):
    """load hunyuan video model

    Args:
        args (dict): model args
        in_channels (int): input channels number
        out_channels (int): output channels number
        factor_kwargs (dict): factor kwargs

    Returns:
        model (nn.Module): The hunyuan video model
    """

    if args.model == "HYVideo-T/2-inverse":
        model = HYVideoDiffusionTransformer_inverse(
            args,
            in_channels=in_channels,
            out_channels=out_channels,
            **HUNYUAN_VIDEO_CONFIG[args.model],
            **factor_kwargs,
        )
        return model

    import pdb; pdb.set_trace()

    if args.model in HUNYUAN_VIDEO_CONFIG.keys():
        model = HYVideoDiffusionTransformer(
            args,
            in_channels=in_channels,
            out_channels=out_channels,
            **HUNYUAN_VIDEO_CONFIG[args.model],
            **factor_kwargs,
        )
        return model
    else:
        raise NotImplementedError()
