import argparse

def parse_args(namespace=None):
    parser = argparse.ArgumentParser(description="StepVideo inference script")

    parser = add_extra_models_args(parser)
    parser = add_denoise_schedule_args(parser)
    parser = add_inference_args(parser)
    parser = add_parallel_args(parser)

    args = parser.parse_args(namespace=namespace)

    return args



def add_extra_models_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(
        title="Extra models args, including vae, text encoders and tokenizers)"
    )

    group.add_argument(
        "--vae_url",
        type=str,
        default='127.0.0.1',
        help="vae url.",
    )
    group.add_argument(
        "--caption_url",
        type=str,
        default='127.0.0.1',
        help="caption url.",
    )

    return parser


def add_denoise_schedule_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(title="Denoise schedule args")

    # Flow Matching
    group.add_argument(
        "--time_shift",
        type=float,
        default=7.0,
        help="Shift factor for flow matching schedulers.",
    )
    group.add_argument(
        "--flow_reverse",
        action="store_true",
        help="If reverse, learning/sampling from t=1 -> t=0.",
    )
    group.add_argument(
        "--flow_solver",
        type=str,
        default="euler",
        help="Solver for flow matching.",
    )

    return parser


def add_inference_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(title="Inference args")

    # ======================== Model loads ========================
    group.add_argument(
        "--model_dir",
        type=str,
        default="./ckpts",
        help="Root path of all the models, including t2v models and extra models.",
    )
    group.add_argument(
        "--model_resolution",
        type=str,
        default="540p",
        choices=["540p"],
        help="Root path of all the models, including t2v models and extra models.",
    )
    group.add_argument(
        "--use-cpu-offload",
        action="store_true",
        help="Use CPU offload for the model load.",
    )

    # ======================== Inference general setting ========================
    group.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for inference and evaluation.",
    )
    group.add_argument(
        "--infer_steps",
        type=int,
        default=50,
        help="Number of denoising steps for inference.",
    )
    group.add_argument(
        "--save_path",
        type=str,
        default="./results",
        help="Path to save the generated samples.",
    )
    group.add_argument(
        "--output_file_name",
        type=str,
        default="",
        help="Name to save the generated samples.",
    )
    group.add_argument(
        "--name_suffix",
        type=str,
        default="",
        help="Suffix for the names of saved samples.",
    )
    group.add_argument(
        "--num_videos",
        type=int,
        default=1,
        help="Number of videos to generate for each prompt.",
    )
    # ---sample size---
    group.add_argument(
        "--num_frames",
        type=int,
        default=102,
        help="How many frames to sample from a video. ",
    )
    group.add_argument(
        "--height",
        type=int,
        default=544,
        help="The height of video sample",
    )
    group.add_argument(
        "--width",
        type=int,
        default=992,
        help="The width of video sample",
    )
    # --- prompt ---
    group.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Prompt for sampling during evaluation.",
    )
    group.add_argument(
        "--first_image_path",
        type=str,
        default='./assets/demo.png',
        help="The reference image path for image-to-video task.",
    )
    group.add_argument("--seed", type=int, default=1234, help="Seed for evaluation.")

    # Classifier-Free Guidance
    group.add_argument(
        "--pos_magic", type=str, default="画面中的主体动作表现生动自然、画面流畅、生动细节、光线统一柔和、超真实动态捕捉、大师级运镜、整体不变形、超高清、画面稳定、逼真的细节、专业级构图、超细节、清晰。", help="Positive magic prompt for sampling."
    )
    group.add_argument(
        "--neg_magic", type=str, default="动画、模糊、变形、毁容、低质量、拼贴、粒状、标志、抽象、插图、计算机生成、扭曲、动作不流畅、面部有褶皱、表情僵硬、畸形手指", help="Negative magic prompt for sampling."
    )
    group.add_argument(
        "--cfg_scale", type=float, default=9.0, help="Classifier free guidance scale."
    )
    group.add_argument(
        "--motion_score", type=float, default=5, help="Score to control the motion level of the video."
    )


    return parser


def add_parallel_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(title="Parallel args")

    # ======================== Model loads ========================
    group.add_argument(
        "--ulysses_degree",
        type=int,
        default=8,
        help="Ulysses degree.",
    )
    group.add_argument(
        "--ring_degree",
        type=int,
        default=1,
        help="Ulysses degree.",
    )

    group.add_argument(
        "--tensor_parallel_degree",
        type=int,
        default=1,
        help="Tensor parallel degree.",
    )
    return parser
