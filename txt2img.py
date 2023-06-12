from modules import paths, sd_samplers
from modules.txt2img import txt2img
from webui import initialize
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# extension/sd-webui-controlnet
from scripts.controlnet_ui.controlnet_ui_group import UiControlNetUnit


# specific sd model load
def sd_model_load(sd_model_fname="v1-5-pruned-emaonly.safetensors"):
    from modules.sd_models import load_model, CheckpointInfo

    sd_model_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "models/Stable-diffusion",
        sd_model_fname,
    )
    checkpoint_info = CheckpointInfo(sd_model_path)
    load_model(checkpoint_info)


# txt2img
def txt2img_wrapper(
    prompt: str = "",
    negative_prompt: str = "",
    steps: int = 20,
    sampler_index: int = 0,
    seed: int = -1,
    subseed: int = -1,
    height: int = 512,
    width: int = 512,
    denoising_strength: float = 0.7,
    hr_scale: float = 2.0,
    hr_upscaler: str = "Latent",
    hr_second_pass_steps: int = 10,
    controlnets: List[UiControlNetUnit] = [],
):
    # sampler index
    print(f"sampler : {sd_samplers.samplers[sampler_index].name}")

    # set max number of controlnets is 4
    _controlnets = list()
    for i in range(4):
        if len(controlnets) > 0:
            _controlnets.append(controlnets.pop(0))
        else:
            _controlnets.append(UiControlNetUnit(enabled=False))

    # default args
    args = [
        "",
        "",
        "",
        [],
        20,
        0,
        False,
        False,
        1,
        1,
        7,
        -1.0,
        -1.0,
        0,
        0,
        0,
        False,
        512,
        512,
        False,
        0.7,
        2,
        "Latent",
        0,
        0,
        0,
        0,
        "",
        "",
        [],
        0,
        _controlnets[0],
        _controlnets[1],
        _controlnets[2],
        _controlnets[3],
        False,
        False,
        "positive",
        "comma",
        0,
        False,
        False,
        "",
        1,
        "",
        [],
        0,
        "",
        [],
        0,
        "",
        [],
        True,
        False,
        False,
        False,
        0,
        None,
        None,
        False,
        None,
        None,
        False,
        None,
        None,
        False,
        None,
        None,
        False,
        50,
    ]

    # update args
    args[1] = prompt
    args[2] = negative_prompt
    args[4] = steps
    args[5] = sampler_index
    args[11] = seed
    args[12] = subseed
    args[17] = height
    args[18] = width

    # Hires. fix
    args[20] = denoising_strength
    args[21] = hr_scale
    args[22] = hr_upscaler
    args[23] = hr_second_pass_steps

    images, _, _, _ = txt2img(*args)
    return images


if __name__ == "__main__":
    # initialize webui modules
    initialize()

    # sd model load
    # sd_model_load(sd_model_fname="v1-5-pruned-emaonly.safetensors")

    # sample
    images = txt2img_wrapper(prompt="cat")
