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

    # -------------------- DR. MOTION --------------------
    def drmotion_generate(self, uploaded_file, model_choice: str, motion_type: str, master_dna: str) -> Dict[str, Any]:
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
            f"Task: Write a video prompt for motion: '{motion_type}'.\n"
            "Include Physics & Lighting: Cloth simulation, Hair physics, Lighting shifts, Weight.\n"
            "Return JSON: {analysis, physics_logic, final_video_prompt}."
        )
        user_text = f"Master Identity: {master_dna}\nModel: {model_choice}\nMotion: {motion_type}\nGenerate prompt."
        messages = [
            {"role": "system", "content": instructions},
            {"role": "user", "content": [{"type": "text", "text": user_text}, {"type": "image_url", "image_url": {"url": data_url}}]},
        ]
        return self._call_chat_json(messages, max_tokens=1500)

    # -------------------- DIGITAL WARDROBE --------------------
    def wardrobe_fuse_filelike(self, uploaded_file, master_dna: str) -> Dict[str, Any]:
        data_url = self._filelike_to_data_url(uploaded_file)
        instructions = (
            "Analyze outfit image (fabric, cut, texture). IGNORE person.\n"
            "Fuse with Master Face DNA.\n"
            "Generate image prompt.\n"
            "Return JSON: {outfit_description, fused_prompt}."
        )
        user_text = f"MASTER DNA:\n{master_dna}\n\nTask: Wear this outfit.\nOutput JSON."
        messages = [
            {"role": "system", "content": instructions},
            {"role": "user", "content": [{"type": "text", "text": user_text}, {"type": "image_url", "image_url": {"url": data_url}}]},
        ]
        return self._call_chat_json(messages, max_tokens=1500)

    # -------------------- MULTI-ANGLE GRID PLANNER (UPDATED) --------------------
    def multi_angle_planner_filelike(self, uploaded_file, master_dna: str) -> Dict[str, Any]:
        data_url = self._filelike_to_data_url(uploaded_file)
        dna_snippet = (master_dna or "")[:800]
        
        instructions = (
            "You are an expert Virtual Photography Director.\n"
            "Task: Design a 'Multi-Angle Character Sheet' (4x5 grid, 20 slots).\n"
            "1. Create a 'grid_prompt' for the image generator. CRITICAL: You MUST explicitly list the 20 angles in the prompt text (e.g., 'Slot 1: Front, Slot 2: Side...').\n"
            "   ALSO: Instruct the generator to 'Burn visible numbers 1-20 into the corner of each grid slot' so the user can identify them.\n"
            "2. Return a structured list of these 20 angles.\n"
            "Return JSON keys: 'grid_prompt' (string), 'angles' (list of objects with id (1-20), name, description)."
        )

        user_text = (
            f"Character DNA:\n{dna_snippet}\n\n"
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
