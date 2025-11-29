# debate_utils.py
import torch
import os
import math
from PIL import Image
from transformers import AutoModel, AutoTokenizer
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

import torch
import math

def split_model(model_name: str):
    """
    Splits the InternVL model across available GPUs.
    - GPU 0 is dedicated to Vision Transformer, embeddings, and final outputs.
    - All Transformer layers are distributed evenly across the remaining GPUs (1, 2, ...).
    """
    device_map = {}
    world_size = torch.cuda.device_count()

    # Handle edge cases: no GPU or only one GPU
    if world_size == 0:
        return "cpu"
    if world_size == 1:
        return "auto"

    # --- 核心修改 ---

    # 1. 将第一块显卡 (GPU 0) 专门用于处理视觉、嵌入和最终的规范化/输出层
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    # 注意：InternVL2 使用 'language_model.model.embed_tokens'
    # 'tok_embeddings' 是旧版或其他模型的叫法，这里保留 embed_tokens
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.output'] = 0

    # 2. 将语言模型的头部 (lm_head) 放在最后一块卡上，以平衡显存
    last_gpu = world_size - 1
    device_map['language_model.lm_head'] = last_gpu

    # 3. 将所有 Transformer 层均匀地分配到除第一块之外的显卡上 (GPUs 1, 2, ...)
    num_layers = 80  # Default for InternVL2-Llama3-76B
    
    #if '26b' in model_name.lower():
    #    num_layers = 48
    #elif '40b' in model_name.lower():
    #    num_layers = 60

    
    # 计算用于分配层的GPU数量和每块GPU应承载的层数
    gpus_for_layers = list(range(1, world_size))
    num_gpus_for_layers = len(gpus_for_layers)
    
    if num_gpus_for_layers > 0:
        layers_per_gpu = math.ceil(num_layers / num_gpus_for_layers)

        layer_idx = 0
        for gpu_id in gpus_for_layers:
            for _ in range(layers_per_gpu):
                if layer_idx < num_layers:
                    device_map[f'language_model.model.layers.{layer_idx}'] = gpu_id
                    layer_idx += 1

    return device_map

def build_transform(input_size):
    """Builds the image transformation pipeline."""
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    """Finds the closest aspect ratio from a set of target ratios."""
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    """Dynamically preprocesses an image by splitting it into blocks."""
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set((i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if i * j <= max_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    if use_thumbnail and len(processed_images) > 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image_pixels(image_obj: Image.Image, input_size=448, max_num=12):
    """Loads a PIL image and prepares it for the model."""
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image_obj, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    return torch.stack(pixel_values)

def load_model_and_tokenizer(model_path: str):
    """Loads the InternVL model and tokenizer from a local path."""
    print(f"Loading model and tokenizer from local path: {model_path}")
    
    model_name = os.path.basename(model_path.strip('/'))
    device_map = split_model(model_name)
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map=device_map
    ).eval()
    
    print("Model and tokenizer loaded successfully!")
    return model, tokenizer

def run_inference(model: AutoModel, tokenizer: AutoTokenizer, prompt: str, image_object: Image.Image) -> str:
    """Performs a single inference pass with the InternVL model."""
    try:
        pixel_values = load_image_pixels(image_object).to(torch.bfloat16)
        
        # Move pixel_values to the same device as the vision model
        vision_device = next(p.device for p in model.vision_model.parameters())
        pixel_values = pixel_values.to(vision_device)

        generation_config = dict(
            num_beams=1,
            max_new_tokens=4096,
            do_sample=False,
        )
        
        response = model.chat(tokenizer, pixel_values, prompt, generation_config)
        return response.strip()

    except Exception as e:
        import traceback
        print("--- An error occurred during inference ---")
        traceback.print_exc()
        print("-----------------------------------------")
        return f"Error during local model inference: {type(e).__name__} - {e}"