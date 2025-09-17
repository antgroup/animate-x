from hyvideo.text_encoder import TextEncoder
from loguru import logger
import torch

# python prepare_text_embedding/get_embedding.py 

text_encoder = TextEncoder(
    text_encoder_type='llm-i2v',
    max_length=359,
    text_encoder_precision='fp16',
    tokenizer_type='llm-i2v',
    i2v_mode=True,
    prompt_template={'template': '<|start_header_id|>system<|end_header_id|>\n\n<image>\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n', 'crop_start': 36, 'image_emb_start': 5, 'image_emb_end': 581, 'image_emb_len': 576, 'double_return_token_id': 271},
    prompt_template_video={'template': '<|start_header_id|>system<|end_header_id|>\n\n<image>\nDescribe the video by detailing the following aspects according to the reference image: 1. The main content and theme of the video.2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects.3. Actions, events, behaviors temporal relationships, physical movement changes of the objects.4. background environment, light, style and atmosphere.5. camera angles, movements, and transitions used in the video:<|eot_id|>\n\n<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n', 'crop_start': 103, 'image_emb_start': 5, 'image_emb_end': 581, 'image_emb_len': 576, 'double_return_token_id': 271},
    hidden_state_skip_layer=2,
    apply_final_norm=False,
    reproduce=False,
    logger=None,
    device="cuda",
    image_embed_interleave=4
)

# motions = ["clap", "bow", "punch", "dance", "handstand", "wave", "knock door", "lift arm", "open door", "salute"]
# motions = ["backflip", "spin", "squat", "clap", "bow", "punch", "dance", "handstand", "wave", "knock door", "lift arm", "open door", "salute"]
motions = ["pray"]



for motion in motions:
# motion = "clap"

    prompt = [f'{motion} A person {motion}']
    data_type = 'video'
    text_inputs = text_encoder.text2tokens(prompt, data_type=data_type)

    from PIL import Image

    i2v_image_path = '00000.jpg'
    semantic_images = [Image.open(i2v_image_path).convert('RGB')]


    prompt_outputs = text_encoder.encode(
        text_inputs, data_type=data_type, semantic_images=semantic_images, device="cuda"
    )
    prompt_embeds = prompt_outputs.hidden_state

    torch.save(prompt_embeds[0][144:], f'text_embedings/{motion}_a_person_{motion}_prompt_embeds.pt')

    print(prompt_embeds.shape)
    attention_mask = prompt_outputs.attention_mask

    effective_condition_sequence_length = attention_mask.sum(dim=1, dtype=torch.int)
    print(effective_condition_sequence_length)