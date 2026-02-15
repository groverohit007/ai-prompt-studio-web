import base64
import json
from typing import Any, Dict, List

from openai import OpenAI


class OpenAIService:
    """
    Streamlit-ready OpenAI service.
    Includes: Cloner, PerfectCloner, Multi-Angle, Wardrobe, DrMotion (Standard + Product Review), Poser, Captions, Prompter.
    SAFE MODE: Removes specific body-measurement triggers from analysis instructions to prevent API refusals.
    """

    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    # -------------------- DR. MOTION (VIDEO) --------------------
    
    def drmotion_generate(self, uploaded_file, model_choice: str, motion_type: str, emotion: str, master_dna: str) -> Dict[str, Any]:
        """Standard single-clip generation with Emotion injection."""
        data_url = self._filelike_to_data_url(uploaded_file)
        
        model_guides = {
            "Kling 1.5": "Focus on 'high quality', '8k', camera orbit, texture realism.",
            "Veo 2 / Sora": "Focus on physics consistency, fluid dynamics, lighting interaction.",
            "Luma Dream Machine": "Focus on 'cinematic', 'keyframe', start/end state.",
            "Runway Gen-3 Alpha": "Focus on 'structure preservation', 'smooth motion', speed/intensity."
        }
        guide = model_guides.get(model_choice, "Focus on realistic motion and physics.")
        
        instructions = (
            f"You are Dr. Motion, expert in {model_choice}.\nStyle Guide: {guide}\n"
            f"Task: Write a video prompt for motion: '{motion_type}' with Emotion: '{emotion}'.\n"
            "CRITICAL - ACTING & MICRO-EXPRESSIONS:\n"
            "   - Do not just say the emotion name. Describe the face.\n"
            "   - If 'Happy': Mention 'crinkling eyes (Duchenne smile)', 'shoulders relaxing'.\n"
            "   - If 'Serious': Mention 'focused gaze', 'firm posture', 'minimal blinking'.\n"
            "   - Include 'Natural Pauses': hesitation before moving, taking a breath.\n"
            "Include Physics: Cloth simulation, Hair physics, Lighting shifts.\n"
            "Return JSON: {analysis, physics_logic, acting_notes, final_video_prompt}."
        )
        
        user_text = (
            f"Master Identity: {master_dna}\n"
            f"Model: {model_choice}\n"
            f"Motion: {motion_type}\n"
            f"Target Emotion: {emotion}\n"
            "Generate prompt."
        )
        
        messages = [
            {"role": "system", "content": instructions},
            {"role": "user", "content": [{"type": "text", "text": user_text}, {"type": "image_url", "image_url": {"url": data_url}}]},
        ]
        return self._call_chat_json(messages, max_tokens=1500)

    def drmotion_product_review(self, uploaded_file, product_info: str, language: str, emotion: str, master_dna: str) -> Dict[str, Any]:
        """
        Generates a 2-part sequence (16s total) for a product review with specific Emotional Tone.
        """
        data_url = self._filelike_to_data_url(uploaded_file)
        
        instructions = (
            "You are an expert AI Commercial Director.\n"
            "Task: Create a cohesive 16-second 'Product Review' sequence split into two 8-second clips.\n"
            f"TONE/EMOTION: {emotion.upper()}.\n"
            "1. Analyze the input image.\n"
            f"2. Write a script in '{language}' that strictly matches the '{emotion}' tone (e.g., if 'High Energy', use short punchy words. If 'Casual', use slang/fillers like 'um', 'actually').\n"
            "3. Create TWO distinct video prompts (Clip A and Clip B).\n"
            "   - Clip A (0-8s): Hook. Focus on Facial Expressions. Include specific acting cues (e.g., 'gasps in delight', 'raises eyebrow skeptically').\n"
            "   - Clip B (8-16s): Demo. Focus on Product. Match lighting of Clip A.\n"
            "Return JSON keys: 'script', 'clip_1_prompt', 'clip_2_prompt', 'continuity_notes'."
        )

        user_text = (
            f"Master Identity: {master_dna}\n"
            f"Product Details: {product_info}\n"
            f"Script Language: {language}\n"
            f"Target Emotion: {emotion}\n"
            "Generate the 2-part video sequence plan."
        )

        messages = [
            {"role": "system", "content": instructions},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ]
        return self._call_chat_json(messages, max_tokens=2000)

    # -------------------- DIGITAL WARDROBE --------------------
    def wardrobe_fuse_filelike(self, uploaded_file, master_dna: str) -> Dict[str, Any]:
        data_url = self._filelike_to_data_url(uploaded_file)
        instructions = (
            "Analyze outfit image (fabric, cut, texture, color). IGNORE the person/body.\n"
            "Fuse this outfit description with the user's locked 'Master Face DNA'.\n"
            "Generate a final image prompt.\n"
            "Return JSON: {outfit_description, fused_prompt}."
        )
        user_text = f"MASTER DNA:\n{master_dna}\n\nTask: Wear this outfit.\nOutput JSON."
        messages = [
            {"role": "system", "content": instructions},
            {"role": "user", "content": [{"type": "text", "text": user_text}, {"type": "image_url", "image_url": {"url": data_url}}]},
        ]
        return self._call_chat_json(messages, max_tokens=1500)

    # -------------------- MULTI-ANGLE GRID PLANNER (SAFE) --------------------
    def multi_angle_planner_filelike(self, uploaded_file, master_dna: str) -> Dict[str, Any]:
        data_url = self._filelike_to_data_url(uploaded_file)
        safe_dna_snippet = (master_dna or "")[:200] 
        
        instructions = (
            "You are an expert Virtual Photography Director.\n"
            "Task: Design a 'Multi-Angle Character Sheet' (4x5 grid, 20 slots).\n"
            "1. Create a 'grid_prompt' for the image generator. CRITICAL: You MUST explicitly list the 20 angles in the prompt text (e.g., 'Slot 1: Front, Slot 2: Side...').\n"
            "   ALSO: Instruct the generator to 'Burn visible numbers 1-20 into the corner of each grid slot'.\n"
            "2. Return a structured list of these 20 angles.\n"
            "Return JSON keys: 'grid_prompt' (string), 'angles' (list of objects with id (1-20), name, description)."
        )

        user_text = (
            f"Character Context: {safe_dna_snippet}\n"
            "Task: Plan 20 distinct camera angles (Low, High, Dutch, Profile, Back, etc.).\n"
            "Output JSON."
        )

        messages = [
            {"role": "system", "content": instructions},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ]
        return self._call_chat_json(messages, max_tokens=2500)

    def build_physics_prompt(self, master_dna: str, angle_data: Dict[str, Any]) -> str:
        angle_name = angle_data.get("name", "Unknown Angle")
        angle_desc = angle_data.get("description", "")
        physics_block = (
            "LIGHTING & PHYSICS (PBR):\n"
            "- Physically Based Rendering (PBR), Raytraced GI.\n"
            "- Subsurface Scattering (SSS) on skin.\n"
            "- Fresnel reflections, Volumetric lighting.\n"
            "- Realistic cast shadows, Ambient Occlusion."
        )
        full_prompt = (
            f"{master_dna.strip()}\n\n"
            f"ANGLE: {angle_name}\nDESC: {angle_desc}\n\n"
            f"{physics_block}\n\n"
            "NEGATIVE: flat, baked lighting, cartoon, bad anatomy, blurry."
        )
        return full_prompt

    # -------------------- CAPTIONS --------------------
    def captions_generate_filelike(self, uploaded_file, style: str = "Engaging", language: str = "English") -> Dict[str, Any]:
        data_url = self._filelike_to_data_url(uploaded_file)
        instructions = "Analyze image. Write ONE Instagram caption with emojis + EXACTLY 4 hashtags. Return JSON: {caption, hashtags}."
        user_content = f"Style: {style}\nLanguage: {language}"
        messages = [
            {"role": "system", "content": instructions},
            {"role": "user", "content": [{"type": "text", "text": user_content}, {"type": "image_url", "image_url": {"url": data_url}}]},
        ]
        data = self._call_chat_json(messages, max_tokens=600)
        hashtags = data.get("hashtags") or []
        if isinstance(hashtags, list): hashtags = [str(h) for h in hashtags[:4]]
        return {"caption": data.get("caption", ""), "hashtags": hashtags}

    # -------------------- CLONER (SAFE MODE) --------------------
    def cloner_analyze_filelike(self, uploaded_file, master_dna: str) -> Dict[str, Any]:
        data_url = self._filelike_to_data_url(uploaded_file)
        instructions = (
            "Analyze the image's pose, lighting, camera angle, and background style.\n"
            "Do NOT analyze body measurements or specific biometrics.\n"
            "Return valid JSON with keys: full_prompt, negative_prompt.\n"
            "Construct the 'full_prompt' by combining the provided MASTER DNA with your analysis of the scene."
        )
        user_text = f"MASTER DNA:\n{master_dna}\n\nAnalyze the scene/lighting/pose of this image and merge it with the DNA."
        messages = [
            {"role": "system", "content": instructions},
            {"role": "user", "content": [{"type": "text", "text": user_text}, {"type": "image_url", "image_url": {"url": data_url}}]},
        ]
        return self._call_chat_json(messages, max_tokens=1000)

    # -------------------- PERFECT CLONER --------------------
    def perfectcloner_analyze_filelike(self, uploaded_file, master_dna: str, identity_lock: bool = True) -> Dict[str, Any]:
        data_url = self._filelike_to_data_url(uploaded_file)
        instructions = "Analyze details (camera, lighting). Return JSON: recreation_prompt, negative_prompt, notes."
        user_text = f"Identity Lock: {identity_lock}\nDNA: {master_dna}\nAnalyze."
        messages = [
            {"role": "system", "content": instructions},
            {"role": "user", "content": [{"type": "text", "text": user_text}, {"type": "image_url", "image_url": {"url": data_url}}]},
        ]
        return self._call_chat_json(messages, max_tokens=1500)

    # -------------------- PROMPTER --------------------
    def prompter_build(self, master_dna: str, fields: Dict[str, str]) -> str:
        parts = [
            master_dna.strip(), "", "PROMPT:",
            f"Pose: {fields.get('pose', '')}",
            f"Attire: {fields.get('attire', '')}",
            f"Camera: {fields.get('camera_angle', '')} | {fields.get('camera_lens', '')}",
            f"Lighting: {fields.get('lighting', '')}",
            f"Background: {fields.get('background', '')}",
            f"Jewellery: {fields.get('jewellery', '')}", "",
            "PHYSICS & REALISM:",
            "- Physically Based Rendering (PBR), realistic shadows, natural skin texture",
            "- Shot on iPhone 17, f/16 look", "",
            "Negative prompt: blurry, bad anatomy, text, watermark"
        ]
        return "\n".join(parts)

    # -------------------- POSER --------------------
    def poser_variations_filelike(self, uploaded_file, master_dna: str, pose_style: str) -> Dict[str, Any]:
        data_url = self._filelike_to_data_url(uploaded_file)
        instructions = "Create 5 pose variations. Return JSON: {prompts: [{pose_name, pose_description, facial_expression}], scene_lock: string}."
        user_text = f"Style: {pose_style}\nReference DNA: {master_dna}\nAnalyze image."
        messages = [
            {"role": "system", "content": instructions},
            {"role": "user", "content": [{"type": "text", "text": user_text}, {"type": "image_url", "image_url": {"url": data_url}}]}
        ]
        return self._call_chat_json(messages)

    # -------------------- HELPERS --------------------
    def _filelike_to_data_url(self, uploaded_file) -> str:
        content = uploaded_file.getvalue()
        mime = getattr(uploaded_file, "type", "image/jpeg") or "image/jpeg"
        b64 = base64.b64encode(content).decode("utf-8")
        return f"data:{mime};base64,{b64}"

    def _sanitize_json_text(self, s: str) -> str:
        if not s: return s
        if s.startswith("```json"): s = s[7:]
        if s.startswith("```"): s = s[3:]
        if s.endswith("```"): s = s[:-3]
        return s.strip()

    def _call_chat_json(self, messages: list, max_tokens: int = 1000) -> Dict[str, Any]:
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
            return json.loads(self._sanitize_json_text(resp.choices[0].message.content))
        except Exception as e:
            print(f"❌ OPENAI ERROR: {e}") 
            if hasattr(e, 'response'):
                print(f"Response: {e.response}")
            return {}
