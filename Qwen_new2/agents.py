from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from debate_utils import run_inference
from PIL import Image

# 严格保留原始定义，不做任何修改
hate_speech_definition = "Any kind of communication [...], that attacks or uses pejorative or discriminatory language with reference to a person or a group on the basis of who they are, in other words, based on their religion, ethnicity, nationality, race, colour, descent, gender or other identity factor."

class BaseAgent:
    def __init__(self, model: Qwen2VLForConditionalGeneration, processor: AutoProcessor, system_prompt: str = ""):
        self.model = model
        self.processor = processor
        self.history = []
        if system_prompt:
            self.history.append({"role": "system", "content": system_prompt})

    def reset_history(self):
        self.history = [{"role": "system", "content": self.history[0]["content"]}] if self.history and self.history[0]["role"] == "system" else []

    def chat(self, user_content: list) -> str:
        self.history.append({"role": "user", "content": user_content})
        response = run_inference(self.model, self.processor, self.history)
        self.history.append({"role": "assistant", "content": response})
        return response

    def one_shot_chat(self, user_content: list) -> str:
        # One-shot: build fresh messages with system + user only
        messages = []
        if self.history and self.history[0]["role"] == "system":
            messages.append(self.history[0])
        messages.append({"role": "user", "content": user_content})
        return run_inference(self.model, self.processor, messages).strip()


# -----------------------------
# ✅ V23: VisualAnalyzer — 返回完整 Q&A 过程（适配 messages 格式）
# -----------------------------
class VisualAnalyzer(BaseAgent):
    def __init__(self, model: Qwen2VLForConditionalGeneration, processor: AutoProcessor):
        system_prompt = "You are an objective visual analyst. You will analyze an image through a series of brief Q&A."
        super().__init__(model, processor, system_prompt)

    def run_analysis_chain(self, image_object: Image.Image) -> dict:
        # Q0: Holistic non-text description
        q0_prompt = "Objectively describe only the visual elements of this image(excluding text). no subjective feelings."
        ans_holistic = self.one_shot_chat([{"type": "image", "image": image_object}, {"type": "text", "text": q0_prompt}])

        # Q1: Human presence
        q1_prompt = "Task: Classify the image. Does it contain human subjects? Respond with ONLY one word: 'Yes' or 'No'."
        ans_subject = self.one_shot_chat([{"type": "image", "image": image_object}, {"type": "text", "text": q1_prompt}])
        has_human = "Yes" in ans_subject or "yes" in ans_subject.lower()

        # Q2/Q3 or Q2b
        if has_human:
            q2_prompt = "Question 2: Describe perceived race, gender, and age."
            ans_demo = self.one_shot_chat([{"type": "image", "image": image_object}, {"type": "text", "text": q2_prompt}])
            
            q3_prompt = "Question 3: Describe body type, clothing, and actions."
            ans_appearance = self.one_shot_chat([{"type": "image", "image": image_object}, {"type": "text", "text": q3_prompt}])
            
            ans_non_human = "N/A"
        else:
            q2b_prompt = "Question 2b: Describe the main non-human subject."
            ans_non_human = self.one_shot_chat([{"type": "image", "image": image_object}, {"type": "text", "text": q2b_prompt}])
            
            ans_demo = "N/A"
            ans_appearance = "N/A"

        # Q4: Background
        q4_prompt = "Question 4: Describe the background."
        ans_setting = self.one_shot_chat([{"type": "image", "image": image_object}, {"type": "text", "text": q4_prompt}])

        # Summary fusion (uses history — now via chat)
        q_summary_prompt = f"""
        Based on the analysis we just performed:
        - Initial description: "{ans_holistic}"
        - Q&A Details: Q1(Human: {ans_subject}), Q2/Q3(Details: {ans_demo}, {ans_appearance}), Q4(background: {ans_setting}).

        Fuse these details (Q1-Q4) into the initial description (Q0).
        Respond with ONLY the final, fused description.
        """
        visual_summary_text = self.chat([{"type": "image", "image": image_object}, {"type": "text", "text": q_summary_prompt}])

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
# ✅ V23: CulturalAnalyzer — 返回完整 Q&A + category-guided（适配 messages 格式）
# -----------------------------
class CulturalAnalyzer(BaseAgent):
    def __init__(self, model: Qwen2VLForConditionalGeneration, processor: AutoProcessor, country: str):
        system_prompt = (
            f"You are a cultural expert specializing in {country}. "
            "You will analyze a meme's cultural aspects based on its visual content, caption, and local context in {country}. "
            "Keep answers **very concise, ideally in a single sentence**."
        )
        self.country = country
        super().__init__(model, processor, system_prompt)

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
        raw_q1 = self.one_shot_chat([{"type": "image", "image": image_object}, {"type": "text", "text": q1_text}])
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

        ans_c1 = self.one_shot_chat([{"type": "image", "image": image_object}, {"type": "text", "text": make_guided_question("Social Norms")}])
        ans_c2 = self.one_shot_chat([{"type": "image", "image": image_object}, {"type": "text", "text": make_guided_question("Minority Group Sensitivity")}])
        ans_c3 = self.one_shot_chat([{"type": "image", "image": image_object}, {"type": "text", "text": make_guided_question("Historical/Political Context")}])
        ans_c4 = self.one_shot_chat([{"type": "image", "image": image_object}, {"type": "text", "text": make_guided_question("Cultural Stereotypes")}])

        # Summary (with history via chat)
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
        cultural_summary_text = self.chat([{"type": "image", "image": image_object}, {"type": "text", "text": summary_prompt}])

        return {
            "q1_raw_response": raw_q1,
            "visual_category": visual_category,
            "q2_social_norms": ans_c1,
            "q3_minority_sensitivity": ans_c2,
            "q4_historical_context": ans_c3,
            "q5_cultural_stereotypes": ans_c4,
            "summary": cultural_summary_text  # 保持兼容
        }



class NonHateDebater(BaseAgent):
    def __init__(self, model: Qwen2VLForConditionalGeneration, processor: AutoProcessor):
        system_prompt = (
        f"You are a debater arguing the meme is **non-hateful** under this definition: '{hate_speech_definition}'. "
        "Assess the meme as a **single, coherent communicative act**—integrating the full caption (including multi-part structures like 'A <sep> B'), image content, and cultural context. "
        "Do not isolate, exaggerate, or decontextualize any word, phrase, or visual element outside this unified meaning. "
        "In your initial argument, address exactly four aspects—caption literal meaning, image surface content, image-caption relationship, and cultural context—each in one sentence only."
        )
        super().__init__(model, processor, system_prompt)

    def generate_initial_argument(self, visual_desc: str, cultural_context: str, caption: str, image_object: Image.Image) -> str:
        user_text = (
            f"Full caption (treat as an integrated phrase): \"{caption}\". "
            f"Image surface content: \"{visual_desc}\". "
            f"Image-caption relationship: How do they jointly convey meaning? "
            f"Cultural context: \"{cultural_context}\". "
            "For each of the four aspects, provide exactly one sentence explaining why the **unified meme** does not attack or use pejorative/discriminatory language against any identity-based group."
        )
        user_content = [{"type": "image", "image": image_object}, {"type": "text", "text": user_text}]
        return self.chat(user_content)


class HateDebater(BaseAgent):
    def __init__(self, model: Qwen2VLForConditionalGeneration, processor: AutoProcessor):
        system_prompt = (
            f"You are a debater arguing the meme is **hateful** under this definition: '{hate_speech_definition}'. "
            "Assess the meme as a **single, coherent communicative act**—integrating the full caption (including multi-part structures like 'A <sep> B'), image content, and cultural context. "
            "Do not isolate, exaggerate, or decontextualize any word, phrase, or visual element outside this unified meaning. "
            "In your initial argument, address exactly four aspects—caption literal meaning, image surface content, image-caption relationship, and cultural context—each in one sentence only."
        )
        super().__init__(model, processor, system_prompt)

    def generate_initial_argument(self, visual_desc: str, cultural_context: str, caption: str, image_object: Image.Image) -> str:
        user_text = (
            f"Full caption (treat as an integrated phrase): \"{caption}\". "
            f"Image surface content: \"{visual_desc}\". "
            f"Image-caption relationship: How do they jointly convey meaning? "
            f"Cultural context: \"{cultural_context}\". "
            "For each of the four aspects, provide exactly one sentence explaining why the **unified meme** attacks or uses pejorative/discriminatory language against an identity-based group."
        )
        user_content = [{"type": "image", "image": image_object}, {"type": "text", "text": user_text}]
        return self.chat(user_content)


class Judge(BaseAgent):
    def __init__(self, model: Qwen2VLForConditionalGeneration, processor: AutoProcessor):
        self.model = model
        self.processor = processor

    def render_judgment(self, debate_transcript: str, country: str, caption: str, visual_summary_text: str, cultural_summary_text: str, image_object: Image.Image) -> str:
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

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [{"type": "image", "image": image_object}, {"type": "text", "text": user_prompt}]}
        ]
        return run_inference(self.model, self.processor, messages)

