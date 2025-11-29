# run_all_ablations.py
import io
import os
import json
import argparse
import pandas as pd
from PIL import Image
from tqdm import tqdm
from debate_utils import load_model_and_processor
from agents import VisualAnalyzer, CulturalAnalyzer, NonHateDebater, HateDebater, Judge
from transformers import LlavaOnevisionForConditionalGeneration, AutoProcessor
import re


# ✅ 更鲁棒的 JSON 解析（匹配 Non-hate / Hate，忽略大小写）
def parse_judgment(raw_response: str) -> dict:
    try:
        match = re.search(r'\b(Non-hate|Hate)\b', raw_response, re.IGNORECASE)
        if match:
            label = match.group(1).capitalize()  # 统一为 "Non-hate" 或 "Hate"
            return {"Label": label, "RawResponse": raw_response}
        return {"Label": "Parsing Error", "Reason": "Failed to match 'Non-hate' or 'Hate'.", "RawResponse": raw_response}
    except Exception as e:
        return {"Label": "Parsing Error", "Reason": f"Error: {str(e)}", "RawResponse": raw_response}


# ✅ 新增：根据 mode 精确控制输入摘要（优先级：_no_both > _no_visual/_no_cultural > full）
def get_summaries_for_mode(mode: str, visual_summary: str, cultural_summary: str):
    """
    返回应实际使用的 visual_summary 和 cultural_summary。
    保证 _no_both 严格双空；避免子串误匹配风险。
    """
    if "_no_both" in mode:
        return "", ""
    elif "_no_visual" in mode:
        return "", cultural_summary
    elif "_no_cultural" in mode:
        return visual_summary, ""
    else:
        return visual_summary, cultural_summary


# ========================
# Precompute visual & cultural analyses per meme per country
# ========================
def precompute_analyses(
    model: LlavaOnevisionForConditionalGeneration,
    processor: AutoProcessor,
    analysis_data_list,
    df,
    country: str
) -> dict:
    cache = {}
    for analysis_info in tqdm(analysis_data_list, desc=f"Pre-analyzing ({country})", leave=False):
        meme_id = int(analysis_info['meme_id'])

        records = df[df['Meme ID'] == meme_id].to_dict('records')
        en_record = next((r for r in records if r['Language'] == 'en'), records[0] if records else None)
        if en_record is None:
            continue
        pil_image = Image.open(io.BytesIO(en_record['image']['bytes'])).convert('RGB')
        caption_en = en_record['Caption']

        try:
            visual_analyzer = VisualAnalyzer(model, processor)
            visual_res = visual_analyzer.run_analysis_chain(pil_image)
            visual_summary = visual_res["summary"]
            del visual_analyzer

            cultural_analyzer = CulturalAnalyzer(model, processor, country)
            cultural_res = cultural_analyzer.run_analysis_chain(caption_en, visual_summary, pil_image)
            cultural_summary = cultural_res["summary"]
            del cultural_analyzer

            cache[(meme_id, country)] = {
                "visual_summary": visual_summary,
                "cultural_summary": cultural_summary,
                "visual_analysis": visual_res,
                "cultural_analysis": cultural_res,
                "caption": caption_en,
                "image": pil_image
            }
        except Exception as e:
            tqdm.write(f"⚠️ Pre-analysis failed for {meme_id} in {country}: {e}")
            continue
    return cache


# ========================
# Run a single ablation experiment for one meme
# ========================
def run_single_experiment(
    mode: str,
    country: str,
    model: LlavaOnevisionForConditionalGeneration,
    processor: AutoProcessor,
    visual_summary: str,
    cultural_summary: str,
    caption: str,
    image: Image.Image,
    rounds: int = 1
) -> dict:
    # ✅ 核心修正：统一获取应使用的摘要（防止 _no_both 失效）
    vs_used, cs_used = get_summaries_for_mode(mode, visual_summary, cultural_summary)

    result = {
        "mode": mode,
        "meme_id": None,
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
            # ✅ 关键修正：Judge 使用 vs_used / cs_used（而非原始 full 摘要）
            raw_response = judge.render_judgment(
                dummy_transcript, country, caption, vs_used, cs_used, image
            )
            ruling = parse_judgment(raw_response)
            result["final_ruling"] = ruling
        except Exception as e:
            result["final_ruling"] = {
                "Label": "Error",
                "Reason": str(e),
                "RawResponse": locals().get("raw_response", "")
            }
        return result

    # 🔷 Debate modes
    try:
        non_hate_debater = NonHateDebater(model, processor)
        hate_debater = HateDebater(model, processor)
        judge = Judge(model, processor)

        # Round 1: Debaters use vs_used / cs_used
        arg_nh1 = non_hate_debater.generate_initial_argument(vs_used, cs_used, caption, image)
        arg_h1 = hate_debater.generate_initial_argument(vs_used, cs_used, caption, image)

        transcript_lines = [
            f"Round 1:",
            f"Non-Hate Debater: {arg_nh1}",
            f"Hate Debater: {arg_h1}"
        ]

        if rounds >= 2:
            arg_nh2 = non_hate_debater.generate_rebuttal(arg_h1, image)
            arg_h2 = hate_debater.generate_rebuttal(arg_nh1, image)
            transcript_lines.extend([
                "", "Round 2:",
                f"Non-Hate Debater (rebuttal): {arg_nh2}",
                f"Hate Debater (rebuttal): {arg_h2}"
            ])
            result.update({
                "round_1_non_hate": arg_nh1,
                "round_1_hate": arg_h1,
                "round_2_non_hate_rebuttal": arg_nh2,
                "round_2_hate_rebuttal": arg_h2,
            })
        else:
            result.update({
                "round_1_non_hate": arg_nh1,
                "round_1_hate": arg_h1
            })

        transcript = "\n".join(transcript_lines)

        # ✅ 关键修正：Judge 使用 vs_used / cs_used（避免信息泄露！）
        raw_response = judge.render_judgment(
            transcript, country, caption, vs_used, cs_used, image
        )
        ruling = parse_judgment(raw_response)
        result["final_ruling"] = ruling

    except Exception as e:
        raw_response = locals().get('raw_response', '')
        result["final_ruling"] = {
            "Label": "Error",
            "Reason": str(e),
            "RawResponse": raw_response
        }
    return result


# ========================
# Main entry
# ========================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="【Meme 辩论实验系统 — 8 种消融实验】")
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--analysis_dir', type=str, required=True)
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--test_single', action='store_true', help="仅运行第一个 meme")
    parser.add_argument('--test_index', type=int, help="指定 meme ID 调试")
    parser.add_argument('--rounds', type=int, default=1, choices=[1, 2], help="辩论轮数")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading Qwen2-VL model and dataset...")
    model, processor = load_model_and_processor(args.model_path)
    df = pd.read_parquet(args.dataset_path)
    df['Meme ID'] = df['Meme ID'].astype(int)
    print("✅ Model and dataset loaded.")

    # ✅ 8 种消融模式（与目标一致）
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

        country_match = re.search(r'analysis_(\w+)\.json', filename)
        if not country_match:
            continue
        country = country_match.group(1)
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

            print(f"\n{'='*40}\n🌍 Country: {country} | Memes: {len(analysis_data)} | Rounds: {args.rounds}\n{'='*40}")

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

            # Save: one file per ablation mode per country
            print(f"\n💾 Saving {country} results (8 ablations) → {args.output_dir}")
            for mode, results in all_results.items():
                if results:
                    out_path = os.path.join(args.output_dir, f"ablation_{mode}_{country}.json")
                    with open(out_path, 'w', encoding='utf-8') as f:
                        json.dump(results, f, indent=2, ensure_ascii=False)
                    print(f"✅ {mode}: {len(results)} memes → {os.path.basename(out_path)}")

            # 🔍 Debug sample
            if args.test_single or args.test_index is not None:
                mid = meme_keys[0][0]
                print(f"\n🔍 Sample Comparison (meme_id={mid}):")
                for mode in ['debate_full', 'debate_no_visual', 'debate_no_cultural', 'debate_no_both']:
                    res = all_results[mode][0]
                    label = res['final_ruling'].get('Label', 'N/A')
                    vs_used = res.get('visual_summary_used', '')[:50].replace('\n', ' ')
                    cs_used = res.get('cultural_summary_used', '')[:50].replace('\n', ' ')
                    print(f"  {mode:20}: {label:10} | VS: {repr(vs_used[:30]+'...' if len(vs_used)>30 else vs_used)} | CS: {repr(cs_used[:30]+'...' if len(cs_used)>30 else cs_used)}")
                test_mode_processed = True
                break

        except Exception as e:
            import traceback
            print(f"❌ Country {country} failed: {e}")
            traceback.print_exc()

    print(f"\n🎉 All 8 ablation experiments completed! Results saved to: {args.output_dir}")