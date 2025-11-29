# agents.py
from transformers import AutoModel, AutoTokenizer
from debate_utils import run_inference
from PIL import Image

# 仇恨言论定义（供提示词使用）
hate_speech_definition = "Any kind of communication [...], that attacks or uses pejorative or discriminatory language with reference to a person or a group on the basis of who they are, in other words, based on their religion, ethnicity, nationality, race, colour, descent, gender or other identity factor."

class BaseAgent:
    def __init__(self, model: AutoModel, tokenizer: AutoTokenizer, system_prompt: str = ""):
        self.model = model
        self.tokenizer = tokenizer
        self.system_prompt = f"{system_prompt}\n\n" if system_prompt else ""
        self.history = self.system_prompt

    def reset_history(self):
        self.history = self.system_prompt

    def chat(self, user_text: str, image_object: Image.Image) -> str:
        prompt_to_add = f"USER: <image>\n{user_text}" if "<image>" not in self.history else f"USER: {user_text}"
        prompt = self.history + prompt_to_add
        full_prompt = prompt + "\n\nASSISTANT: "
        response = run_inference(self.model, self.tokenizer, full_prompt, image_object)
        self.history += f"{prompt_to_add}\n\nASSISTANT: {response}\n\n"
        return response

    def one_shot_chat(self, user_text: str, image_object: Image.Image) -> str:
        prompt = f"{self.system_prompt}USER: <image>\n{user_text}\n\nASSISTANT: "
        response = run_inference(self.model, self.tokenizer, prompt, image_object)
        return response.strip()


# -----------------------------
# ✅ V23: VisualAnalyzer — 返回完整 Q&A 过程
# -----------------------------
class VisualAnalyzer(BaseAgent):
    def __init__(self, model: AutoModel, tokenizer: AutoTokenizer):
        system_prompt = "You are an objective visual analyst. You will analyze an image through a series of brief Q&A."
        super().__init__(model, tokenizer, system_prompt)

    def run_analysis_chain(self, image_object: Image.Image) -> dict:
        # Q0: Holistic non-text description
        q0_prompt = "Objectively describe only the visual elements of this image(excluding text). no subjective feelings."
        ans_holistic = self.one_shot_chat(q0_prompt, image_object).strip()

        # Q1: Human presence
        q1_prompt = "Task: Classify the image. Does it contain human subjects? Respond with ONLY one word: 'Yes' or 'No'."
        ans_subject = self.one_shot_chat(q1_prompt, image_object).strip()
        has_human = "Yes" in ans_subject or "yes" in ans_subject.lower()

        # Q2/Q3 or Q2b
        if has_human:
            q2_prompt = "Question 2: Describe perceived race, gender, and age."
            ans_demo = self.one_shot_chat(q2_prompt, image_object).strip()
            
            q3_prompt = "Question 3: Describe body type, clothing, and actions."
            ans_appearance = self.one_shot_chat(q3_prompt, image_object).strip()
            
            ans_non_human = "N/A"
        else:
            q2b_prompt = "Question 2b: Describe the main non-human subject."
            ans_non_human = self.one_shot_chat(q2b_prompt, image_object).strip()
            
            ans_demo = "N/A"
            ans_appearance = "N/A"

        # Q4: Background
        q4_prompt = "Question 4: Describe the background."
        ans_setting = self.one_shot_chat(q4_prompt, image_object).strip()

        # Summary fusion (uses history)
        q_summary_prompt = f"""
        Based on the analysis we just performed:
        - Initial description: "{ans_holistic}"
        - Q&A Details: Q1(Human: {ans_subject}), Q2/Q3(Details: {ans_demo}, {ans_appearance}), Q4(background: {ans_setting}).

        Fuse these details (Q1-Q4) into the initial description (Q0).
        Respond with ONLY the final, fused description.
        """

        visual_summary_text = self.chat(q_summary_prompt, image_object).strip()

        return {
            "q0_initial_description": ans_holistic,
            "q1_human_presence": ans_subject,
            "q2_human_demographics": ans_demo,
            "q3_human_appearance_action": ans_appearance,
            "q2b_non_human_subject": ans_non_human,
            "q4_background": ans_setting,
            "has_human": has_human,
            "summary": visual_summary_text  # 保持兼容
        }


# -----------------------------
# ✅ V23: CulturalAnalyzer — 返回完整 Q&A + category-guided
# -----------------------------
class CulturalAnalyzer(BaseAgent):
    def __init__(self, model: AutoModel, tokenizer: AutoTokenizer, country: str):
        system_prompt = (
            f"You are a cultural expert specializing in {country}. "
            "You will analyze a meme's cultural aspects based on its visual content, caption, and local context in {country}. "
            "Keep answers **very concise, ideally in a single sentence**."
        )
        self.country = country
        super().__init__(model, tokenizer, system_prompt)

    def run_analysis_chain(self, caption: str, visual_summary_text: str, image_object: Image.Image) -> dict:
        base_prompt = (
            f"[Task] Analyze the cultural implications of the following meme.\n"
            f"[Country] {self.country}\n"
            f"[Caption] {caption}\n"
            f"[Visual Summary] {visual_summary_text}\n\n"
            "Based *only* on this information and the image, answer the following question precisely."
        )

        # Q1: Category classification
        q1_text = (
            f"{base_prompt}\n"
            "Q1: Based on the visual summary and image, which single category does the image most relate to? "
            "Choose from: (a) Ethnicity, (b) Political Issues, (c) Religion, (d) Nationality, (e) LGBTQ+. "
            "Respond with only the category name."
        )
        raw_q1 = self.one_shot_chat(q1_text, image_object).strip()
        category_map = {
            "Ethnicity": "Ethnicity", "Political Issues": "Political Issues",
            "Religion": "Religion", "Nationality": "Nationality", "LGBTQ+": "LGBTQ+",
            "(a)": "Ethnicity", "(b)": "Political Issues", "(c)": "Religion",
            "(d)": "Nationality", "(e)": "LGBTQ+", "a": "Ethnicity", "b": "Political Issues",
            "c": "Religion", "d": "Nationality", "e": "LGBTQ+",
        }
        visual_category = category_map.get(raw_q1, raw_q1)

        # Guided Q2–Q5
        def make_guided_question(topic: str) -> str:
            return (
                f"{base_prompt}\n"
                f"Note: The image is primarily categorized as [{visual_category}]. "
                f"Q: Briefly analyze [{topic}]. NEVER speculate or invent facts."
            )

        ans_c1 = self.one_shot_chat(make_guided_question("Social Norms"), image_object).strip()
        ans_c2 = self.one_shot_chat(make_guided_question("Minority Group Sensitivity"), image_object).strip()
        ans_c3 = self.one_shot_chat(make_guided_question("Historical/Political Context"), image_object).strip()
        ans_c4 = self.one_shot_chat(make_guided_question("Cultural Stereotypes"), image_object).strip()

        # Summary
        summary_prompt = f"""
        We have analyzed a meme categorized as: **{visual_category}**.

        - Q2 (Social Norms in {visual_category} context): {ans_c1}
        - Q3 (Minority Sensitivity in {visual_category} context): {ans_c2}
        - Q4 (Historical/Political Context in {visual_category} context): {ans_c3}
        - Q5 (Cultural Stereotypes in {visual_category} context): {ans_c4}

        Consolidate into a concise, bulleted cultural risk/insight summary.
        Start with "Visual Category: {visual_category}".
        Limit: ≤80 words total.
        """
        cultural_summary_text = self.chat(summary_prompt, image_object).strip()

        return {
            "q1_raw_response": raw_q1,
            "visual_category": visual_category,
            "q2_social_norms": ans_c1,
            "q3_minority_sensitivity": ans_c2,
            "q4_historical_context": ans_c3,
            "q5_cultural_stereotypes": ans_c4,
            "summary": cultural_summary_text  # 保持兼容
        }


# -----------------------------
# 辩论 Agent (保持不变，仅依赖 summary 字段)
# -----------------------------
class NonHateDebater(BaseAgent):
    """
    A debater arguing for the non-hateful side, using InternVL.
    """
    def __init__(self, model: AutoModel, tokenizer: AutoTokenizer):
        system_prompt = (
            f"You are a debater arguing the meme is **non-hateful** under this definition: '{hate_speech_definition}'. "
            "Assess the meme as a **single, coherent communicative act**—integrating the full caption (including multi-part structures like 'A <sep> B'), image content, and cultural context. "
            "Do not isolate, exaggerate, or decontextualize any word, phrase, or visual element outside this unified meaning. "
            "In your initial argument, address exactly four aspects—caption literal meaning, image surface content, image-caption relationship, and cultural context—each in one sentence only."
        )
        super().__init__(model, tokenizer, system_prompt)

    def generate_initial_argument(self, visual_desc: str, cultural_context: str, caption: str, image_object: Image.Image) -> str:
        user_text = (
            f"Full caption (treat as an integrated phrase): \"{caption}\". "
            f"Image surface content: \"{visual_desc}\". "
            f"Image-caption relationship: How do they jointly convey meaning? "
            f"Cultural context: \"{cultural_context}\". "
            "For each of the four aspects, provide exactly one sentence explaining why the **unified meme** does not attack or use pejorative/discriminatory language against any identity-based group."
        )
        # For InternVL, the <image> token is a placeholder for where the model should consider the image.
        return self.chat(f"<image>\n{user_text}", image_object)

    def generate_rebuttal(self, opponent_argument: str, image_object: Image.Image) -> str:
        user_text = (
            f"Opponent's argument: \"{opponent_argument}\". "
            "If their reasoning focuses on a fragment of the caption or image while ignoring the integrated meaning of the full meme, point out this is **out of context**. "
            "Emphasize that meaning emerges from the **combined elements**, not isolated parts."
        )        
        return self.chat(f"<image>\n{user_text}", image_object)


class HateDebater(BaseAgent):
    """
    A debater arguing for the hateful side, using InternVL.
    """
    def __init__(self, model: AutoModel, tokenizer: AutoTokenizer):
        system_prompt = (
            f"You are a debater arguing the meme is **hateful** under this definition: '{hate_speech_definition}'. "
            "Assess the meme as a **single, coherent communicative act**—integrating the full caption (including multi-part structures like 'A <sep> B'), image content, and cultural context. "
            "Do not isolate, exaggerate, or decontextualize any word, phrase, or visual element outside this unified meaning. "
            "In your initial argument, address exactly four aspects—caption literal meaning, image surface content, image-caption relationship, and cultural context—each in one sentence only."
        )
        super().__init__(model, tokenizer, system_prompt)

    def generate_initial_argument(self, visual_desc: str, cultural_context: str, caption: str, image_object: Image.Image) -> str:
        user_text = (
            f"Full caption (treat as an integrated phrase): \"{caption}\". "
            f"Image surface content: \"{visual_desc}\". "
            f"Image-caption relationship: How do they jointly convey meaning? "
            f"Cultural context: \"{cultural_context}\". "
            "For each of the four aspects, provide exactly one sentence explaining why the **unified meme** attacks or uses pejorative/discriminatory language against an identity-based group."
        )
        return self.chat(f"<image>\n{user_text}", image_object)

    def generate_rebuttal(self, opponent_argument: str, image_object: Image.Image) -> str:
        user_text = (
            f"Opponent's argument: \"{opponent_argument}\". "
            "If you maintain the meme is hateful, you must demonstrate that the **coherent meaning** of the full caption, image, and context—not a fragmented excerpt—constitutes an attack or discriminatory expression based on identity. "
            "Avoid selective interpretation; address the meme as a complete communicative act."
        )        
        return self.chat(f"<image>\n{user_text}", image_object)


class Judge(BaseAgent):
    def __init__(self, model: AutoModel, tokenizer: AutoTokenizer):
        super().__init__(model, tokenizer)

    def render_judgment(self, debate_transcript: str, country: str, caption: str, visual_summary_text: str, cultural_summary_text: str, image_object: Image.Image) -> str:
        system_prompt = (
            f"You are a judge from **{country}** applying only this definition: '{hate_speech_definition}'. "
            "Assess the meme as a **complete communicative unit**: interpret multi-part captions as integrated statements, and combine them with the image and cultural context. "
            "Reject any argument that relies on taking words or images out of their full context. "
            "Output only a JSON object with keys 'Label' and 'Reason'."
        )
        user_prompt = (
            f"Debate Transcript:\n{debate_transcript}\n\n"
            f"Full caption (integrated meaning): \"{caption}\"\n\n"
            "Render your judgment in this exact format:\n"
            '{"Label": "Hate or Non-hate", "Reason": "One-sentence justification based on all evidence."}'
        )

        full_prompt = f"{system_prompt}\n\nUSER: <image>\n{user_prompt}\n\nASSISTANT: "
        return run_inference(self.model, self.tokenizer, full_prompt, image_object)