from modules import paths, sd_samplers
from modules.txt2img import txt2img
from modules.img2img import img2img
from modules.sd_models import load_model, CheckpointInfo
import modules
from webui import initialize
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import PIL.Image
import numpy as np
from diffusers.utils import load_image

# extension/sd-webui-controlnet
from scripts.controlnet_ui.controlnet_ui_group import UiControlNetUnit
from scripts.external_code import get_models, ControlMode, get_modules, ResizeMode


# initialize sd models
initialize()

# initialize scripts
modules.scripts.scripts_txt2img.initialize_scripts(is_img2img=False)
modules.scripts.scripts_img2img.initialize_scripts(is_img2img=True)


# specific sd model load
def sd_model_load(sd_model_fname="v1-5-pruned-emaonly.safetensors"):
    sd_model_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "models/Stable-diffusion",
        sd_model_fname,
    )
    checkpoint_info = CheckpointInfo(sd_model_path)
    load_model(checkpoint_info)


def depth_controlnet(
    control_image: PIL.Image,
    is_depth_map=True,
    model=None,
    module=None,
    weight: float = 1.0,
    processor_res=512,
    control_mode=ControlMode.BALANCED.value,
    resize_mode=ResizeMode.INNER_FIT.value,
):
    # depth controlnet
    model = [s for s in get_models(True) if "depth" in s][0] if model is None else model
    module = (
        [s for s in get_modules(True) if "depth_midas" in s][0]
        if module is None
        else module
    )

    if is_depth_map:
        # [Case 1] control image : depth map image
        controlnet = UiControlNetUnit(
            enabled=True,
            image=None,
            generated_image=control_image,
            use_preview_as_input=True,
            model=model,
            control_mode=control_mode,
            resize_mode=resize_mode,
            processor_res=processor_res,
            weight=weight,
        )
    else:
        # [Case 2] control image : original image (unprocessed)
        control_image = np.array(control_image.convert("RGB"))
        fake_mask = np.zeros(
            (control_image.shape[0], control_image.shape[1], 4), dtype=np.uint8
        )
        fake_mask[:, :, 3] = 255
        image = {"image": control_image, "mask": fake_mask}
        controlnet = UiControlNetUnit(
            enabled=True,
            image=image,
            model=model,
            module=module,
            control_mode=control_mode,
            resize_mode=resize_mode,
            processor_res=processor_res,
            weight=weight,
        )
    return controlnet


def inpaint_controlnet(
    model=None,
    module=None,
    weight: float = 1.0,
    control_mode=ControlMode.BALANCED.value,
    resize_mode=ResizeMode.INNER_FIT.value,
):
    # Inpaint controlnet
    model = (
        [s for s in get_models(True) if "inpaint" in s][0] if model is None else model
    )
    module = (
        [s for s in get_modules(True) if "inpaint_only" in s][0]
        if module is None
        else module
    )

    controlnet = UiControlNetUnit(
        enabled=True,
        image=None,
        model=model,
        module=module,
        control_mode=control_mode,
        resize_mode=resize_mode,
        weight=weight,
    )
    return controlnet


def reference_controlnet(
    control_image: PIL.Image,
    module=None,
    style_fidelity=0.5,
    weight: float = 1.0,
    control_mode=ControlMode.BALANCED.value,
    resize_mode=ResizeMode.INNER_FIT.value,
):
    # reference controlnet
    module = (
        [s for s in get_modules(True) if "reference_only" in s][0]
        if module is None
        else module
    )

    control_image = np.array(control_image.convert("RGB"))
    fake_mask = np.zeros(
        (control_image.shape[0], control_image.shape[1], 4), dtype=np.uint8
    )
    fake_mask[:, :, 3] = 255
    image = {"image": control_image, "mask": fake_mask}

    controlnet = UiControlNetUnit(
        enabled=True,
        image=image,
        module=module,
        control_mode=control_mode,
        resize_mode=resize_mode,
        weight=weight,
        threshold_a=style_fidelity,
    )
    return controlnet


def txt2img_wrapper(
    prompt: str = "",
    negative_prompt: str = "",
    steps: int = 20,
    sampler_index: int = 0,
    seed: int = -1,
    subseed: int = -1,
    height: int = 512,
    width: int = 512,
    enable_hr: bool = False,
    denoising_strength: float = 0.7,
    hr_scale: float = 2.0,
    hr_upscaler: str = "Latent",
    hr_second_pass_steps: int = 0,
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

    # prepare args
    args = [
        "",
        prompt,
        negative_prompt,
        [],
        steps,
        sampler_index,
        False,
        False,
        1,
        1,
        7,
        seed,
        subseed,
        0,
        0,
        0,
        False,
        height,
        width,
        enable_hr,
        denoising_strength,
        hr_scale,
        hr_upscaler,
        hr_second_pass_steps,
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

    images, _, _, _ = txt2img(*args)
    return images


def img2img_inpaint_wrapper(
    init_img: PIL.Image.Image = None,  # (RGBA)
    mask: PIL.Image.Image = None,  # (RGBA) mask region: 255, rest: 0
    prompt: str = "",
    negative_prompt: str = "",
    steps: int = 20,
    sampler_index: int = 0,
    seed: int = -1,
    subseed: int = -1,
    height: int = 512,
    width: int = 512,
    denoising_strength: float = 0.75,
    resize_mode: int = 0,
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

    # prepare args
    args = [
        "",
        2,
        prompt,
        negative_prompt,
        [],
        None,
        None,
        {"image": init_img.convert("RGBA"), "mask": mask.convert("RGBA")},
        None,
        None,
        None,
        None,
        steps,
        sampler_index,
        4,
        0,
        1,
        False,
        False,
        1,
        1,
        7,
        1.5,
        denoising_strength,
        seed,
        subseed,
        0,
        0,
        0,
        False,
        0,
        height,
        width,
        1,
        resize_mode,
        0,
        32,
        0,
        "",
        "",
        "",
        [],
        0,
        _controlnets[0],
        _controlnets[1],
        _controlnets[2],
        _controlnets[3],
        "<ul>\n<li><code>CFG Scale</code> should be 2 or lower.</li>\n</ul>\n",
        True,
        True,
        "",
        "",
        True,
        50,
        True,
        1,
        0,
        False,
        4,
        0.5,
        "Linear",
        "None",
        '<p style="margin-bottom:0.75em">Recommended settings: Sampling Steps: 80-100, Sampler: Euler a, Denoising strength: 0.8</p>',
        128,
        8,
        ["left", "right", "up", "down"],
        1,
        0.05,
        128,
        4,
        0,
        ["left", "right", "up", "down"],
        False,
        False,
        "positive",
        "comma",
        0,
        False,
        False,
        "",
        '<p style="margin-bottom:0.75em">Will upscale the image by the selected scale factor; use width and height sliders to set tile size</p>',
        64,
        0,
        2,
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

    images, _, _, _ = img2img(*args)
    return images


if __name__ == "__main__":
    # Prerequisite & execution command
    """
    git clone https://github.com/adif-ai/stable-diffusion-webui.git
    cd stable-diffusion-webui
    conda create -n webui python=3.8
    conda activate webui
    pip install -r requirements.txt
    pip install git+https://github.com/huggingface/diffusers
    cd extensions/sd-webui-controlnet/models
    wget https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11f1p_sd15_depth.pth
    wget https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_inpaint.pth
    cd ../../..
    python launch.py
    python sd_webui_modules.py
    """

    # Specific Stable Diffusion model Load (./models/Stable-diffusion)
    # sd_model_load(sd_model_fname="v1-5-pruned-emaonly.safetensors")

    # load images
    init_image = load_image(
        "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main/stable_diffusion_inpaint/boy.png"
    )

    mask_image = load_image(
        "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main/stable_diffusion_inpaint/boy_mask.png"
    )

    depth_image = init_image

    reference_image = load_image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d3/Robert_Downey%2C_Jr._2012.jpg/1200px-Robert_Downey%2C_Jr._2012.jpg"
    )

    # img2img inpaint with multi controlnet
    controlnets = [
        depth_controlnet(
            control_image=init_image,
            is_depth_map=False,
        ),
        inpaint_controlnet(),
    ]

    images = img2img_inpaint_wrapper(
        prompt="sunglass boy",
        init_img=init_image,
        mask=mask_image,
        controlnets=controlnets,
        seed=1,
    )
    images[0].save("img2img_depth+inpaint.png")

    # img2img inpaint with multi controlnet
    controlnets = [
        inpaint_controlnet(),
        depth_controlnet(
            control_image=init_image,
            is_depth_map=False,
        ),
        reference_controlnet(control_image=reference_image, style_fidelity=0.2),
    ]

    images = img2img_inpaint_wrapper(
        prompt="sunglass boy",
        init_img=init_image,
        mask=mask_image,
        controlnets=controlnets,
        seed=1,
    )
    images[0].save("img2img_depth+inpaint+reference.png")

    # txt2img with multi controlnet
    controlnets = [
        depth_controlnet(
            control_image=init_image,
            is_depth_map=False,
        ),
        reference_controlnet(control_image=reference_image, style_fidelity=0.1),
    ]

    images = txt2img_wrapper(
        prompt="sunglass boy",
        controlnets=controlnets,
        seed=1,
    )
    images[0].save("txt2img_depth+reference.png")
