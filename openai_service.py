import base64
import json
from typing import Any, Dict, List

from openai import OpenAI


class OpenAIService:
    """
    Streamlit-ready OpenAI service.
    Includes: Cloner, PerfectCloner, Multi-Angle, Wardrobe, DrMotion, Poser, Captions, Prompter.
    """

    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    # -------------------- DR. MOTION (NEW) --------------------
    def drmotion_generate(self, uploaded_file, model_choice: str, motion_type: str, master_dna: str) -> Dict[str, Any]:
        data_url = self._filelike_to_data_url(uploaded_file)
        
        # Model-Specific Best Practices (Hidden Knowledge)
        model_guides = {
            "Kling 1.5": "Focus on 'high quality', '8k', and specific camera moves like 'camera orbit'. Mention texture realism.",
            "Veo 2 / Sora": "Focus on physics consistency, fluid dynamics, lighting interaction, and temporal coherence.",
            "Luma Dream Machine": "Focus on 'cinematic', 'keyframe', and start/end state descriptions.",
            "Runway Gen-3 Alpha": "Focus on 'structure preservation', 'smooth motion', and specific speed/intensity adjectives."
        }
        
        guide = model_guides.get(model_choice, "Focus on realistic motion and physics.")

        instructions = (
            f"You are Dr. Motion, an expert AI Video Prompt Engineer specializing in {model_choice}.\n"
            f"STYLE GUIDE FOR {model_choice}: {guide}\n\n"
            "TASK:\n"
            "1. Analyze the uploaded image to understand the subject and setting.\n"
            f"2. Write a specialized video generation prompt to animate this subject performing: '{motion_type}'.\n"
            "3. CRITICAL: Include specific 'Physics & Lighting Logics':\n"
            "   - Cloth Simulation: How fabric moves/wrinkles with the motion.\n"
            "   - Hair Physics: How hair reacts to gravity/wind/movement.\n"
            "   - Lighting: How shadows shift and highlights travel across surfaces during movement.\n"
            "   - Weight: Describe the weight of footsteps or gestures.\n\n"
            "Return JSON keys: 'analysis', 'physics_logic', 'final_video_prompt'."
        )

        user_text = (
            f"Master Identity: {master_dna}\n"
            f"Target Model: {model_choice}\n"
            f"Target Motion: {motion_type}\n"
            "Generate the video prompt."
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
        return self._call_chat_json(messages, max_tokens=1500)

    # -------------------- DIGITAL WARDROBE --------------------
    def wardrobe_fuse_filelike(self, uploaded_file, master_dna: str) -> Dict[str, Any]:
        data_url = self._filelike_to_data_url(uploaded_file)
        instructions = (
            "You are an expert AI Fashion Stylist.\n"
            "1. Analyze the uploaded image and extract the 'Outfit DNA' (fabric, cut, texture, color). IGNORE the person; focus on clothing.\n"
            "2. Combine this Outfit DNA with the user's locked 'Master Face DNA'.\n"
            "3. Generate a final image prompt featuring the Master DNA character wearing this outfit.\n"
            "Return JSON keys: 'outfit_description', 'fused_prompt'."
        )
        user_text = f"MASTER DNA:\n{master_dna}\n\nTask: Wear this outfit.\nOutput JSON."
        messages = [
            {"role": "system", "content": instructions},
            {"role": "user", "content": [{"type": "text", "text": user_text}, {"type": "image_url", "image_url": {"url": data_url}}]},
        ]
        return self._call_chat_json(messages, max_tokens=1500)

    # -------------------- MULTI-ANGLE GRID PLANNER --------------------
    def multi_angle_planner_filelike(self, uploaded_file, master_dna: str) -> Dict[str, Any]:
        data_url = self._filelike_to_data_url(uploaded_file)
        dna_snippet = (master_dna or "")[:800]
        instructions = (
            "Analyze character. Design a 'Multi-Angle Character Sheet' plan.\n"
            "1. Create 'grid_prompt' for a 4x5 grid image (20 slots) of this character in 20 DIFFERENT angles.\n"
            "2. List those 20 angles.\n"
            "Return JSON: 'grid_prompt' (string), 'angles' (list of {id, name, description})."
        )
        user_text = f"Character DNA:\n{dna_snippet}\n\nPlan 20 angles."
        messages = [
            {"role": "system", "content": instructions},
            {"role": "user", "content": [{"type": "text", "text": user_text}, {"type": "image_url", "image_url": {"url": data_url}}]},
        ]
        return self._call_chat_json(messages, max_tokens=2000)

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

    # -------------------- CLONER --------------------
    def cloner_analyze_filelike(self, uploaded_file, master_dna: str) -> Dict[str, Any]:
        data_url = self._filelike_to_data_url(uploaded_file)
        instructions = "Analyze image. Return keys: full_prompt, negative_prompt. full_prompt MUST start with MASTER DNA."
        user_text = f"MASTER DNA:\n{master_dna}\n\nAnalyze this image."
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
            print(f"Error: {e}")
            return {}
