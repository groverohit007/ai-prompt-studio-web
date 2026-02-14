import base64
import json
from typing import Any, Dict, List

from openai import OpenAI


class OpenAIService:
    """
    Streamlit-ready OpenAI service:
      - Uses standard Chat Completions API.
      - Includes Cloner, PerfectCloner, Multi-Angle, Poser, Captions, and Wardrobe.
    """

    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    # -------------------- DIGITAL WARDROBE (NEW) --------------------
    def wardrobe_fuse_filelike(self, uploaded_file, master_dna: str) -> Dict[str, Any]:
        data_url = self._filelike_to_data_url(uploaded_file)
        
        instructions = (
            "You are an expert AI Fashion Stylist.\n"
            "1. Analyze the uploaded image and extract the 'Outfit DNA' (fabric, cut, texture, color, neckline, fit, accessories). IGNORE the person in the reference; focus only on the clothing.\n"
            "2. Combine this Outfit DNA with the user's locked 'Master Face DNA'.\n"
            "3. Generate a final image generation prompt that features the Master DNA character wearing this exact outfit.\n"
            "Return JSON keys: 'outfit_description', 'fused_prompt'."
        )

        user_text = (
            f"MASTER DNA (Face/Body Lock):\n{master_dna}\n\n"
            "Task: Create a prompt where this character is wearing the outfit from the image.\n"
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
        return self._call_chat_json(messages, max_tokens=1500)

    # -------------------- MULTI-ANGLE GRID PLANNER --------------------
    def multi_angle_planner_filelike(self, uploaded_file, master_dna: str) -> Dict[str, Any]:
        data_url = self._filelike_to_data_url(uploaded_file)
        dna_snippet = (master_dna or "")[:800]
        
        instructions = (
            "You are an expert Virtual Photography Director.\n"
            "Analyze the character in the image and design a 'Multi-Angle Character Sheet' plan.\n"
            "1. Create a 'grid_prompt' that would generate a single 4x5 grid image (20 slots) of this character in 20 DIFFERENT camera angles/poses, numbered 1-20.\n"
            "2. Create a detailed list of those 20 angles.\n"
            "Return JSON with keys: 'grid_prompt' (string), 'angles' (list of objects with id (1-20), name, description)."
        )

        user_text = (
            f"Character DNA:\n{dna_snippet}\n\n"
            "Task: Plan 20 distinct camera angles (e.g., Low Angle, Top-Down, Dutch Tilt, Profile, Back View, Close-up, etc.) for this character.\n"
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
        return self._call_chat_json(messages, max_tokens=2000)

    def build_physics_prompt(self, master_dna: str, angle_data: Dict[str, Any]) -> str:
        angle_name = angle_data.get("name", "Unknown Angle")
        angle_desc = angle_data.get("description", "")

        physics_block = (
            "LIGHTING & PHYSICS (PBR) REQUIREMENTS:\n"
            "- Physically Based Rendering (PBR) workflow with raytraced global illumination.\n"
            "- Accurate Subsurface Scattering (SSS) on skin.\n"
            "- Fresnel reflections on eyes and moist surfaces.\n"
            "- Volumetric lighting with Tyndall effect.\n"
            "- Realistic cast shadows with accurate penumbra.\n"
            "- Ambient Occlusion (AO) in crevices.\n"
            "- High Dynamic Range (HDR) exposure simulation."
        )

        full_prompt = (
            f"{master_dna.strip()}\n\n"
            "LOCKED ANGLE / POSE:\n"
            f"Camera Angle: {angle_name}\n"
            f"Pose Description: {angle_desc}\n\n"
            f"{physics_block}\n\n"
            "NEGATIVE PROMPT:\n"
            "flat lighting, baked lighting, cartoonish, cel shaded, 2D, bad anatomy, deformed, extra limbs, blurry, low res."
        )
        return full_prompt

    # -------------------- CAPTIONS --------------------
    def captions_generate_filelike(self, uploaded_file, style: str = "Engaging", language: str = "English") -> Dict[str, Any]:
        data_url = self._filelike_to_data_url(uploaded_file)
        instructions = (
            "You are a social media caption writer.\n"
            "Analyze the image and write ONE Instagram caption that is detailed, engaging, and uses lots of emojis.\n"
            "Also provide EXACTLY 4 hashtags.\n"
            "Return JSON: {caption: string, hashtags: [string]}."
        )
        user_content = f"Style: {style}\nLanguage: {language}\nReturn JSON."

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

    # -------------------- PERFECT CLONER (SIMPLIFIED FOR ROBUSTNESS) --------------------
    def perfectcloner_analyze_filelike(self, uploaded_file, master_dna: str, identity_lock: bool = True) -> Dict[str, Any]:
        data_url = self._filelike_to_data_url(uploaded_file)
        instructions = "Analyze image details (camera, lighting, composition). Return JSON keys: recreation_prompt, negative_prompt, notes."
        user_text = f"Identity Lock: {identity_lock}\nDNA: {master_dna}\nAnalyze."
        messages = [
            {"role": "system", "content": instructions},
            {"role": "user", "content": [{"type": "text", "text": user_text}, {"type": "image_url", "image_url": {"url": data_url}}]},
        ]
        return self._call_chat_json(messages, max_tokens=1500)

    # -------------------- PROMPTER --------------------
    def prompter_build(self, master_dna: str, fields: Dict[str, str]) -> str:
        parts = [
            master_dna.strip(),
            "",
            "PROMPT:",
            f"Pose: {fields.get('pose', '')}",
            f"Attire: {fields.get('attire', '')}",
            f"Camera: {fields.get('camera_angle', '')} | {fields.get('camera_lens', '')}",
            f"Lighting: {fields.get('lighting', '')}",
            f"Background: {fields.get('background', '')}",
            f"Jewellery: {fields.get('jewellery', '')}",
            "",
            "PHYSICS & REALISM:",
            "- Physically Based Rendering (PBR), realistic shadows, natural skin texture",
            "- Shot on iPhone 17, f/16 look",
            "",
            "Negative prompt: blurry, bad anatomy, text, watermark"
        ]
        return "\n".join(parts)

    # -------------------- POSER --------------------
    def poser_variations_filelike(self, uploaded_file, master_dna: str, pose_style: str) -> Dict[str, Any]:
        data_url = self._filelike_to_data_url(uploaded_file)
        instructions = "Create 5 distinct pose variations based on the image mood. Return JSON: {prompts: [{pose_name, pose_description, facial_expression}], scene_lock: string}."
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
