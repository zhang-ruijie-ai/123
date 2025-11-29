# debate_utils.py
import torch
import os
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, modeling_utils
from qwen_vl_utils import process_vision_info

def load_model_and_processor(model_path: str):
    """从本地路径加载多模态模型和对应的 Processor。"""
    print(f"正在从本地路径加载模型和 Processor: {model_path}")

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True
    ).eval()
    
    print("模型和 Processor 加载成功！")
    return model, processor

def run_inference(model: Qwen2VLForConditionalGeneration, processor: AutoProcessor, messages: list) -> str:
    """
    执行一次完整的、与多模态模型兼容的推理。
    现在 messages 列表内部包含的是图片的文件路径。
    """
    try:
        # 步骤 1: 将聊天记录转换为纯文本提示。
        text_prompt = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(messages)

        # 步骤 3: 将文本和加载好的图片对象列表传给处理器。
        inputs = processor(
            text=[text_prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

        # 步骤 4: 生成回复
        outputs = model.generate(
            **inputs,
            max_new_tokens=4096,
            do_sample=False, 
            top_k=None
        )

        # 步骤 5: 解码新生成的部分
        input_token_len = inputs["input_ids"].shape[1]
        response = processor.decode(outputs[0, input_token_len:], skip_special_tokens=True)
        return response.strip()

    except Exception as e:
        import traceback
        print("--- 推理过程中发生错误 ---")
        traceback.print_exc()
        print("--------------------------")
        return f"本地模型推理时出错: {type(e).__name__} - {e}"