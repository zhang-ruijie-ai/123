import io
import os
import json
import argparse
import pandas as pd
from PIL import Image
from tqdm import tqdm
from debate_utils import load_model_and_processor
from agents import VisualAnalyzer, CulturalAnalyzer, NonHateDebater, HateDebater, Judge
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import torch
import re


# 🔧 标准化：语言 ↔ 国家映射（与下方参考代码一致）
LANGUAGE_TO_COUNTRY = {
    'zh': 'China',
    'en': 'USA',
    'de': 'Germany',
    'es': 'Mexico',
    'hi': 'India'
}
COUNTRY_TO_LANG = {v: k for k, v in LANGUAGE_TO_COUNTRY.items()}


def get_summaries_for_mode(mode: str, visual_summary: str, cultural_summary: str):
    if "_no_both" in mode:
        return "", ""
    elif "_no_visual" in mode:
        return "", cultural_summary
    elif "_no_cultural" in mode:
        return visual_summary, ""
    else:
        return visual_summary, cultural_summary


def precompute_analyses(
    model: Qwen2VLForConditionalGeneration,
    processor: AutoProcessor,
    analysis_data_list: list,
    df: pd.DataFrame,
    country: str
) -> dict:
    """
    返回 cache[(meme_id, country)] = {
        "visual_summary", "cultural_summary",
        "visual_analysis", "cultural_analysis",
        "caption", "image"
    }
    """
    if country not in COUNTRY_TO_LANG:
        tqdm.write(f"⚠️ Skipping unknown country: {country}")
        return {}

    expected_lang = COUNTRY_TO_LANG[country]
    cache = {}

    for analysis_info in tqdm(analysis_data_list, desc=f"Pre-analyzing ({country})", leave=False):
        meme_id = int(analysis_info['meme_id'])

        # ✅ 严格匹配 meme_id + expected language
        records = df[
            (df['Meme ID'] == meme_id) &
            (df['Language'] == expected_lang)
        ].to_dict('records')

        if not records:
            tqdm.write(f"⚠️ No record for Meme ID {meme_id} in {country} ({expected_lang}); skipping.")
            continue

        # Take first match (should be unique)
        record = records[0]
        try:
            image_bytes = record['image']['bytes']
            pil_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        except Exception as e:
            tqdm.write(f"⚠️ Failed to load image for {meme_id} in {country}: {e}")
            continue

        caption = record.get('Caption', '').strip() or analysis_info.get('caption', '').strip()
        if not caption:
            tqdm.write(f"⚠️ Empty caption for {meme_id} in {country}; using placeholder.")
            caption = "[No caption]"

        try:
            # Visual analysis
            visual_analyzer = VisualAnalyzer(model, processor)
            visual_res = visual_analyzer.run_analysis_chain(pil_image)
            visual_summary = visual_res["summary"]
            del visual_analyzer

            # Cultural analysis
            cultural_analyzer = CulturalAnalyzer(model, processor, country)
            cultural_res = cultural_analyzer.run_analysis_chain(caption, visual_summary, pil_image)
            cultural_summary = cultural_res["summary"]
            del cultural_analyzer

            cache[(meme_id, country)] = {
                "visual_summary": visual_summary,
                "cultural_summary": cultural_summary,
                "visual_analysis": visual_res,
                "cultural_analysis": cultural_res,
                "caption": caption,
                "image": pil_image
            }

        except Exception as e:
            tqdm.write(f"⚠️ Pre-analysis failed for {meme_id} in {country}: {e}")
            continue

    return cache


def run_single_experiment(
    mode: str,
    country: str,
    model: Qwen2VLForConditionalGeneration,
    processor: AutoProcessor,
    visual_summary: str,
    cultural_summary: str,
    caption: str,
    image: Image.Image,
    rounds: int = 1
) -> dict:
    vs_used, cs_used = get_summaries_for_mode(mode, visual_summary, cultural_summary)

    result = {
        "mode": mode,
        "country": country,
        "caption": caption,
        "visual_summary_used": vs_used,
        "cultural_summary_used": cs_used
    }

    if "judge" in mode:
        judge = Judge(model, processor)
        dummy_transcript = (
            f"Visual Description: {vs_used}\n"
            f"Cultural Context: {cs_used}\n"
            "(Direct judgment without debate.)"
        )

        try:
            raw_response = judge.render_judgment(dummy_transcript, country, caption, vs_used, cs_used, image)
            # Try robust JSON parsing
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', raw_response, re.DOTALL)
            json_str = json_match.group(1) if json_match else raw_response[raw_response.find('{'):raw_response.rfind('}') + 1]
            if json_str.strip():
                ruling = json.loads(json_str)
            else:
                raise ValueError("No JSON found")
            result["final_ruling"] = ruling
        except Exception as e:
            result["final_ruling"] = {
                "Label": "Error",
                "Reason": raw_response  # keep full context
            }
        return result

    # 🔷 Debate mode
    try:
        non_hate_debater = NonHateDebater(model, processor)
        hate_debater = HateDebater(model, processor)
        judge = Judge(model, processor)

        # Round 1
        arg_nh1 = non_hate_debater.generate_initial_argument(vs_used, cs_used, caption, image)
        arg_h1 = hate_debater.generate_initial_argument(vs_used, cs_used, caption, image)

        transcript_lines = [
            f"Round 1:",
            f"Non-Hate Debater: {arg_nh1}",
            f"Hate Debater: {arg_h1}"
        ]

        saved_args = {
            "round_1_non_hate": arg_nh1,
            "round_1_hate": arg_h1
        }

        # Round 2
        if rounds >= 2:
            arg_nh2 = non_hate_debater.generate_rebuttal(arg_h1, image)
            arg_h2 = hate_debater.generate_rebuttal(arg_nh1, image)
            transcript_lines.extend([
                "", "Round 2:",
                f"Non-Hate Debater (rebuttal): {arg_nh2}",
                f"Hate Debater (rebuttal): {arg_h2}"
            ])
            saved_args.update({
                "round_2_non_hate_rebuttal": arg_nh2,
                "round_2_hate_rebuttal": arg_h2,
            })

        transcript = "\n".join(transcript_lines)

        # Judge
        raw_response = judge.render_judgment(transcript, country, caption, vs_used, cs_used, image)
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', raw_response, re.DOTALL)
        json_str = json_match.group(1) if json_match else raw_response[raw_response.find('{'):raw_response.rfind('}') + 1]
        ruling = json.loads(json_str) if json_str.strip() else {"Label": "Error", "Reason": "Empty JSON"}
        result["final_ruling"] = ruling
        result.update(saved_args)

    except Exception as e:
        raw_response = locals().get('raw_response', str(e))
        result["final_ruling"] = {
            "Label": "FatalError",
            "Reason": raw_response
        }

    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="【Meme 辩论实验系统 — 修正版：精准图像/caption加载】")
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--analysis_dir', type=str, required=True)
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--test_single', action='store_true', help="仅运行每个国家的第一个 meme")
    parser.add_argument('--test_index', type=int, help="指定 meme ID 调试（全局唯一 ID）")
    parser.add_argument('--rounds', type=int, default=1, choices=[1, 2], help="辩论轮数（仅 debate 模式生效）")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading Qwen2-VL model and dataset...")
    model, processor = load_model_and_processor(args.model_path)
    df = pd.read_parquet(args.dataset_path)
    df['Meme ID'] = df['Meme ID'].astype(int)
    print("✅ Model and dataset loaded.")

    ABLATION_MODES = [
        'debate_full',
        'debate_no_visual',
        'debate_no_cultural',
        'debate_no_both',
        'judge_full',
        'judge_no_visual',
        'judge_no_cultural',
        'judge_no_both'
    ]

    analysis_files = sorted(f for f in os.listdir(args.analysis_dir) if f.startswith("analysis_") and f.endswith(".json"))
    test_mode_processed = False

    for filename in tqdm(analysis_files, desc="Countries", unit="country"):
        if args.test_single and test_mode_processed:
            break

        # Extract country name: analysis_China.json → China
        country_match = re.search(r'analysis_([A-Za-z]+)\.json', filename)
        if not country_match:
            tqdm.write(f"⚠️ Skipping invalid filename: {filename}")
            continue
        country = country_match.group(1)

        # ✅ Validate country
        if country not in COUNTRY_TO_LANG:
            tqdm.write(f"⚠️ Unknown country '{country}' in filename '{filename}'; skipping.")
            continue

        analysis_path = os.path.join(args.analysis_dir, filename)

        try:
            with open(analysis_path, 'r', encoding='utf-8') as f:
                analysis_data = json.load(f)
            if not isinstance(analysis_data, list):
                tqdm.write(f"{country}: analysis file not a list, skip.")
                continue

            if args.test_index is not None:
                analysis_data = [info for info in analysis_data if int(info['meme_id']) == args.test_index]
            elif args.test_single:
                analysis_data = analysis_data[:1]

            if not analysis_data:
                tqdm.write(f"{country}: no memes to process.")
                continue

            print(f"\n{'='*40}\n🌍 Country: {country} ({COUNTRY_TO_LANG[country]}) | Memes: {len(analysis_data)} | Rounds: {args.rounds}\n{'='*40}")

            # Precompute (now uses standardized country → lang mapping)
            cache = precompute_analyses(model, processor, analysis_data, df, country)
            if not cache:
                tqdm.write(f"{country}: no valid pre-analysis results.")
                continue

            all_results = {mode: [] for mode in ABLATION_MODES}
            meme_keys = list(cache.keys())

            for (meme_id, _) in tqdm(meme_keys, desc=f"Experiments ({country})", unit="meme", leave=False):
                entry = cache[(meme_id, country)]
                vs = entry["visual_summary"]
                cs = entry["cultural_summary"]
                va = entry["visual_analysis"]
                ca = entry["cultural_analysis"]
                caption = entry["caption"]
                image = entry["image"]

                for mode in ABLATION_MODES:
                    try:
                        res = run_single_experiment(
                            mode=mode,
                            country=country,
                            model=model,
                            processor=processor,
                            visual_summary=vs,
                            cultural_summary=cs,
                            caption=caption,
                            image=image,
                            rounds=args.rounds
                        )
                        full_res = {
                            "meme_id": meme_id,
                            "country": country,
                            "caption": caption,
                            "visual_analysis": va,
                            "visual_summary": vs,
                            "cultural_analysis": ca,
                            "cultural_summary": cs,
                            **res
                        }
                        all_results[mode].append(full_res)
                    except Exception as e:
                        tqdm.write(f"❌ Fatal error in mode {mode} for {meme_id}: {e}")
                        full_res = {
                            "meme_id": meme_id,
                            "country": country,
                            "caption": caption,
                            "visual_analysis": va if 'va' in locals() else {},
                            "visual_summary": vs if 'vs' in locals() else "",
                            "cultural_analysis": ca if 'ca' in locals() else {},
                            "cultural_summary": cs if 'cs' in locals() else "",
                            "mode": mode,
                            "final_ruling": {"Label": "FatalError", "Reason": str(e)}
                        }
                        all_results[mode].append(full_res)

            print(f"\n💾 Saving {country} results (8 ablations) → {args.output_dir}")
            for mode, results in all_results.items():
                if results:
                    out_path = os.path.join(args.output_dir, f"ablation_{mode}_{country}.json")
                    with open(out_path, 'w', encoding='utf-8') as f:
                        json.dump(results, f, indent=2, ensure_ascii=False)
                    print(f"✅ {mode}: {len(results)} memes → {os.path.basename(out_path)}")

            # Debug summary for first meme
            if args.test_single or args.test_index is not None:
                first_meme_id = meme_keys[0][0]
                print(f"\n🔍 Sample Comparison (meme_id={first_meme_id}, country={country}):")
                labels = {}
                for mode in ABLATION_MODES:
                    label = all_results[mode][0]['final_ruling'].get('Label', 'N/A')
                    labels[mode] = label
                for mode in ABLATION_MODES:
                    print(f"  {mode:25}: {labels[mode]}")
                print(f"\n📝 Sample Reason (debate_full):")
                reason = all_results['debate_full'][0]['final_ruling'].get('Reason', 'N/A')
                print(f"  {reason[:150]}...")
                test_mode_processed = True
                break

        except Exception as e:
            import traceback
            print(f"❌ Country {country} failed: {e}")
            traceback.print_exc()

    print(f"\n🎉 All experiments completed! Results saved to: {args.output_dir}")