import torch
from diffusers import FluxPipeline

model_id = "" #you can also use `black-forest-labs/FLUX.1-dev`

pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power

prompt_template = "A {} standing like a human, with only two hind legs on the ground, only one front paw held high in the air and one front paw on one side of the body"

# prompt = "A cat standing, with only two hind legs on the ground and two front paws on either side of the body, like a human being"
# prompt = "A cat is standing sideways to the right of a door, the door is on the left, the cat is on the right, the cat is facing the door, with only two hind legs on the ground and two front paws on either side of the body, like a human being"

# prompt = "A cat standing like a human, with only two hind legs on the ground and two front paws held high in the air, cyberpunk style"
# prompt = "A cat standing like a human, with only two hind legs on the ground, only one front paw held high in the air and one front paw on one side of the body, cyberpunk style"

animals = [
    'Cat', 'Dog', 'Lion', 'Tiger', 'Leopard', 'Elephants',
    'Giraffe', 'Zebra', 'Polar bear', 'Dolphin', 'Crocodile',
    'Fox', 'Orangutan', 'Giant Panda', 'Penguin', 'Turtle',
    'Whales', 'Hippo', 'Squirrels', 'wolf', 'sheep', 'Donkey',
    'Parrot', 'kangaroo', 'Wild boar', 'rabbit', 'koala',
    'Bee', 'Butterfly', 'octopus', 'sea lion', 'Bald Eagle',
    'Bears', 'Sea Turtle', 'Seagulls', 'Fish', 'Lizards',
    'goats', 'Chicken', 'Red panda', 'Reindeer', 'Chimpanzee',
    'Horned horse',
]
seed = 42

for i in range(5):
    current_seed = seed+i
    for animal in animals:

        prompt = prompt_template.format(animal)
        image = pipe(
            prompt,
            output_type="pil",
            num_inference_steps=50, #use a larger number if you are using [dev]
            generator=torch.Generator("cpu").manual_seed(seed),
            height = 360,
            width = 640,
        ).images[0]
        image.save(f"flux/flux/dev/animal_wave/{animal}_{i}.png")