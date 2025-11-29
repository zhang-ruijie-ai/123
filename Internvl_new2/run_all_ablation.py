# run_all_ablations.py
import io
import os
import json
import argparse
import pandas as pd
from PIL import Image
from tqdm import tqdm
from debate_utils import load_model_and_tokenizer, load_image_pixels
from agents import VisualAnalyzer, CulturalAnalyzer, NonHateDebater, HateDebater, Judge 
from transformers import AutoModel, AutoTokenizer
import torch
import re
from typing import Optional, Tuple, Dict, List

hate_speech_definition = "Any kind of communication [...], that attacks or uses pejorative or discriminatory language with reference to a person or a group on the basis of who they are, in other words, based on their religion, ethnicity, nationality, race, colour, descent, gender or other identity factor."


# ✅ 全部 8 种 ablation 模式定义（4 info conditions × 2 rounds）
ABLATION_MODES = [
    'round_0__full_info',
    'round_0__no_visual',
    'round_0__no_cultural',
    'round_0__no_both',
    'round_1__full_info',
    'round_1__no_visual',
    'round_1__no_cultural',
    'round_1__no_both'
]

def run_inference_ablated(model: AutoModel, tokenizer: AutoTokenizer, prompt: str, image_object) -> str:
    """
    适配模型推理接口
    """
    pixel_values = None
    if image_object is not None:
        try:
            device = next(model.parameters()).device
            pixel_values = load_image_pixels(image_object).to(torch.bfloat16).to(device)
        except Exception as e:
            return f"Error during image processing: {e}"
    else:
        # Remove <image> tokens when no image is provided
        prompt = prompt.replace("<image>\n", "").replace("<image>", "")

    try:
        generation_config = dict(num_beams=1, max_new_tokens=4096, do_sample=False)
        response = model.chat(tokenizer, pixel_values, prompt, generation_config)
        return response.strip()
    except Exception as e:
        return f"Error during model inference: {e}"


def parse_judgment(raw_response: str) -> dict:
    """
    从原始响应中解析JSON
    """
    try:
        json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
        if json_match:
            markdown_match = re.search(r'```json\s*(\{.*?\})\s*```', raw_response, re.DOTALL)
            if markdown_match:
                return json.loads(markdown_match.group(1))
            return json.loads(json_match.group(0))
        return {"Label": "Parsing Error", "Reason": "Failed to extract JSON.", "RawResponse": raw_response}
    except json.JSONDecodeError:
        return {"Label": "Parsing Error", "Reason": "Exception during JSON parsing.", "RawResponse": raw_response}


# ===================================================================================
# SECTION 2: ABLATION-AWARE AGENT WRAPPERS (with tokenizer only)
# ===================================================================================

class AblationNonHateDebater(NonHateDebater):
    """
    适配消融实验的NonHateDebater类
    """
    def __init__(self, model, tokenizer):
        super().__init__(model, tokenizer)
        self._model = model
        self._tokenizer = tokenizer

    def chat(self, user_text: str, image_object: Optional[Image.Image]) -> str:
        prompt = self.history + "USER: " + user_text
        full_prompt = prompt + "\n\nASSISTANT: "
        response = run_inference_ablated(self._model, self._tokenizer, full_prompt, image_object)
        self.history += f"USER: {user_text}\n\nASSISTANT: {response}\n\n"
        return response


class AblationHateDebater(HateDebater):
    """
    适配消融实验的HateDebater类
    """
    def __init__(self, model, tokenizer):
        super().__init__(model, tokenizer)
        self._model = model
        self._tokenizer = tokenizer

    def chat(self, user_text: str, image_object: Optional[Image.Image]) -> str:
        prompt = self.history + "USER: " + user_text
        full_prompt = prompt + "\n\nASSISTANT: "
        response = run_inference_ablated(self._model, self._tokenizer, full_prompt, image_object)
        self.history += f"USER: {user_text}\n\nASSISTANT: {response}\n\n"
        return response


class AblationJudge(Judge):
    def __init__(self, model, tokenizer):
        super().__init__(model, tokenizer)
        self._model = model
        self._tokenizer = tokenizer

    def render_judgment(self, debate_transcript: str, country: str, caption: str, image_object: Optional[Image.Image]) -> str:
        system_prompt = (
            f"You are a judge from **{country}** applying only this definition: '{hate_speech_definition}'. "
            "Assess the meme as a **complete communicative unit**: interpret multi-part captions as integrated statements, and combine them with the image and cultural context. "
            "Reject any argument that relies on taking words or images out of their full context. "
            "Output only a JSON object with keys 'Label' and 'Reason'."
        )

        user_prompt = (
            f"Debate Transcript:\n##\n{debate_transcript}\n##\n\n"
            f"Full caption (integrated meaning): \"{caption}\"\n\n"
            "Render your judgment in this exact format:\n"
            '{"Label": "Hate or Non-hate", "Reason": "One-sentence justification based on the unified meaning of the entire meme."}'
        )
        full_prompt = f"{system_prompt}\n\nUSER: <image>\n{user_prompt}\n\nASSISTANT: "
        return run_inference_ablated(self._model, self._tokenizer, full_prompt, image_object)


def run_debate_rounds(
    model, tokenizer,
    visual_desc: str,
    cultural_context: str,
    caption: str,
    image_object,
    num_rounds: int = 1
):
    """
    运行辩论回合
    """
    nh = AblationNonHateDebater(model, tokenizer)
    h = AblationHateDebater(model, tokenizer)

    transcript_parts = []
    log = {}

    # Round 1
    arg_nh = nh.generate_initial_argument(visual_desc, cultural_context, caption, image_object)
    arg_h = h.generate_initial_argument(visual_desc, cultural_context, caption, image_object)
    transcript_parts.append(f"Round 1:\nNon-Hate Debater: {arg_nh}\nHate Debater: {arg_h}")
    log["round_1_non_hate"] = arg_nh
    log["round_1_hate"] = arg_h

    if num_rounds >= 2:
        rebuttal_nh = nh.generate_rebuttal(arg_h, image_object)
        rebuttal_h = h.generate_rebuttal(arg_nh, image_object)
        transcript_parts.append(f"Round 2:\nNon-Hate Debater: {rebuttal_nh}\nHate Debater: {rebuttal_h}")
        log["round_2_non_hate"] = rebuttal_nh
        log["round_2_hate"] = rebuttal_h

    full_transcript = "\n\n".join(transcript_parts)
    return full_transcript, log


def build_round0_transcript(visual_desc: str, cultural_context: str) -> str:
    """
    构建无辩论回合的转录
    """
    return (
        f"Visual Description: {visual_desc}\n"
        f"Cultural Context: {cultural_context}\n"
        "(Direct judgment without debate.)"
    )


def run_ablation_with_shared_analysis(
    ablation_mode: str,
    country: str,
    model: AutoModel,
    tokenizer: AutoTokenizer,
    caption: str,
    visual_summary_text: str,
    cultural_summary_text: str,
    visual_analysis_result: dict,
    cultural_analysis_result: dict,
    image_object: Image.Image,
) -> dict:
    """
    使用预计算的 visual/cultural 分析结果，执行单个 ablation。
    """
    # 解析消融模式
    round_type, info_condition = ablation_mode.split('__', 1)

    # 信息条件映射
    info_configs = {
        'full_info': (visual_summary_text, cultural_summary_text),
        'no_visual': ("", cultural_summary_text),
        'no_cultural': (visual_summary_text, ""),
        'no_both': ("", "")
    }

    if info_condition not in info_configs:
        raise ValueError(f"Unknown info condition: {info_condition}")
    
    vd_use, cc_use = info_configs[info_condition]

    if round_type == 'round_1':
        # 运行1轮辩论
        transcript, log = run_debate_rounds(
            model, tokenizer, vd_use, cc_use, caption, image_object, num_rounds=1
        )
    elif round_type == 'round_0':
        # 直接判断，无辩论
        transcript = build_round0_transcript(vd_use, cc_use)
        log = {
            "round_1_non_hate": "",
            "round_1_hate": ""
        }
    else:
        raise ValueError(f"Unknown round type: {round_type}")

    # Judge 进行最终判断
    judge = AblationJudge(model, tokenizer)
    final_judgment_str = judge.render_judgment(transcript, country, caption, image_object)
    final_ruling_json = parse_judgment(final_judgment_str)

    return {
        "ablation_mode": ablation_mode,
        "info_condition": info_condition,
        "round_type": round_type,
        "round_1_non_hate": log["round_1_non_hate"],
        "round_1_hate": log["round_1_hate"],
        "transcript": transcript,
        "final_ruling": final_ruling_json,
    }


# ========================
# Main
# ========================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="【8合1消融实验】每个meme只做1次visual/cultural分析，8个ablation共享结果")
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--analysis_dir', type=str, required=True)
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--test_single', action='store_true', help="仅运行第一个 meme 并打印调试信息")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("正在加载模型和数据集，请稍候...")
    model, tokenizer = load_model_and_tokenizer(args.model_path)
    df = pd.read_parquet(args.dataset_path)
    df['Meme ID'] = df['Meme ID'].astype(int)
    print("✅ 模型和数据集加载完成。")

    # 语言映射
    language_to_country = {
        'zh': 'China',
        'en': 'USA',
        'de': 'Germany',
        'es': 'Mexico',
        'hi': 'India'
    }
    country_to_lang = {v: k for k, v in language_to_country.items()}

    test_mode_processed = False

    for filename in sorted(os.listdir(args.analysis_dir)):
        if filename.startswith("analysis_") and filename.endswith(".json"):
            if args.test_single and test_mode_processed:
                break

            analysis_file_path = os.path.join(args.analysis_dir, filename)

            try:
                country_match = re.search(r'analysis_(\w+)\.json', filename)
                if not country_match:
                    continue
                country = country_match.group(1)

                if country not in country_to_lang:
                    tqdm.write(f"⚠️ Unknown country: {country}; skipping.")
                    continue
                expected_lang = country_to_lang[country]

                print(f"\n{'='*60}")
                print(f"🌍 处理国家: {country} | 语言: {expected_lang} | 文件: {filename}")
                print(f"🎯 将运行 8 个消融实验（共享分析）")
                print(f"{'='*60}")

                with open(analysis_file_path, 'r', encoding='utf-8') as f:
                    analysis_data = json.load(f)

                # 为当前国家准备 8 个结果列表（按 ablation 分组）
                country_results = {mode: [] for mode in ABLATION_MODES}

                progress_bar = tqdm(analysis_data, desc=f"{country} memes", unit="meme")
                for analysis_info in progress_bar:
                    meme_id = int(analysis_info['meme_id'])
                    progress_bar.set_postfix(meme_id=meme_id)

                    # ✅ 修正：同时按 Meme ID 和语言过滤
                    image_records = df[
                        (df['Meme ID'] == meme_id) &
                        (df['Language'] == expected_lang)
                    ]
                    if image_records.empty:
                        tqdm.write(f"⚠️  未找到 Meme ID {meme_id} 且语言为 {expected_lang} 的记录")
                        continue

                    image_bytes = image_records.iloc[0]['image']['bytes']
                    pil_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
                    caption = analysis_info['caption']

                    visual_analyzer = VisualAnalyzer(model, tokenizer)
                    visual_analysis_result = visual_analyzer.run_analysis_chain(pil_image)
                    del visual_analyzer
                    visual_summary_text = visual_analysis_result["summary"]

                    cultural_analyzer = CulturalAnalyzer(model, tokenizer, country)
                    cultural_analysis_result = cultural_analyzer.run_analysis_chain(
                        caption, visual_summary_text, pil_image
                    )
                    del cultural_analyzer
                    cultural_summary_text = cultural_analysis_result["summary"]

                    # 🚀 对 8 个 ablation 并行计算（共享上述分析结果）
                    for mode in ABLATION_MODES:
                        ablation_result = run_ablation_with_shared_analysis(
                            ablation_mode=mode,
                            country=country,
                            model=model,
                            tokenizer=tokenizer,
                            caption=caption,
                            visual_summary_text=visual_summary_text,
                            cultural_summary_text=cultural_summary_text,
                            visual_analysis_result=visual_analysis_result,
                            cultural_analysis_result=cultural_analysis_result,
                            image_object=pil_image
                        )
                        # 合并基础信息 + ablation-specific
                        full_result = {
                            "meme_id": meme_id,
                            "country": country,
                            "visual_analysis": visual_analysis_result,
                            "visual_summary": visual_summary_text,
                            "cultural_analysis": cultural_analysis_result,
                            "cultural_summary": cultural_summary_text,
                            **ablation_result
                        }
                        country_results[mode].append(full_result)

                    # 🔍 调试：只跑第一个 meme
                    if args.test_single:
                        print(f"\n{'='*50}")
                        print(f"[DEBUG] Meme ID: {meme_id} | Country: {country} | Language: {expected_lang}")
                        print(f"[VISUAL] {visual_summary_text[:100]}...")
                        print(f"[CULTURAL] {cultural_summary_text[:100]}...")
                        for mode in ABLATION_MODES[:2]:  # 只打前2个
                            res = country_results[mode][-1]
                            print(f"[{mode}] → Label: {res['final_ruling'].get('Label', 'N/A')}")
                        print("="*50)
                        test_mode_processed = True
                        break

                # 📤 保存当前国家的结果：每个 ablation 模式一个文件
                print(f"\n💾 正在保存 {country} 的结果到 {args.output_dir} ...")
                for mode, results in country_results.items():
                    if results:
                        out_path = os.path.join(args.output_dir, f"ablation_{mode}_{country}.json")
                        with open(out_path, 'w', encoding='utf-8') as f:
                            json.dump(results, f, indent=4, ensure_ascii=False)
                        print(f"✅ {country} - {mode}: {len(results)} memes → {os.path.basename(out_path)}")

                if args.test_single and test_mode_processed:
                    break

            except Exception as e:
                import traceback
                print(f"\n❌ 处理 {filename} 出错: {e}")
                traceback.print_exc()

    print(f"\n🎉 全部国家的消融实验完成！结果目录: {args.output_dir}")
