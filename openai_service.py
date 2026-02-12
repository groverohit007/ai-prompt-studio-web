import base64
import json
from typing import Any, Dict, List

from openai import OpenAI


class OpenAIService:
    """
    Streamlit-ready OpenAI service:
      - Uses standard Chat Completions API (client.chat.completions.create) for stability.
      - Supports Streamlit UploadedFile (file-like) via *_filelike methods.
      - Robust JSON parsing & repair.
      - Enforces Body structure: Hourglass (36-28-36).
      - POSER returns *compact JSON*.
      - Captions: returns caption + exactly 4 hashtags.
    """

    def __init__(self, api_key: str, model: str = "gpt-4.1-mini"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    # -------------------- CAPTIONS --------------------

    def captions_generate_filelike(
        self,
        uploaded_file,
        style: str = "Engaging",
        language: str = "English",
    ) -> Dict[str, Any]:
        data_url = self._filelike_to_data_url(uploaded_file)
        return self.captions_generate_data_url(data_url, style, language)

    def captions_generate_data_url(
        self,
        data_url: str,
        style: str = "Engaging",
        language: str = "English",
    ) -> Dict[str, Any]:
        instructions = (
            "You are a social media caption writer.\n"
            "Analyze the image and write ONE Instagram caption that is detailed, engaging, and uses lots of emojis.\n"
            "Also provide EXACTLY 4 hashtags that are relevant to the image.\n\n"
            "Return ONLY valid JSON with keys:\n"
            "caption (string), hashtags (array of exactly 4 strings).\n\n"
            "Rules:\n"
            "- Keep it safe-for-work.\n"
            "- No markdown.\n"
            "- Hashtags must start with #.\n"
            "- Language must match the user's choice.\n"
        )

        user_content = (
            f"Caption style: {style}\n"
            f"Language: {language}\n"
            "Guidance:\n"
            "- English: natural Instagram English.\n"
            "- Hindi: Devanagari Hindi.\n"
            "- Hinglish: Hindi + English mix in Roman script.\n"
            "Return JSON only."
        )

        messages = [
            {"role": "system", "content": instructions},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_content},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ]

        data = self._call_chat_json(messages, max_tokens=650)

        caption = (data.get("caption") or "").strip()
        hashtags = data.get("hashtags") or []
        if not isinstance(hashtags, list):
            hashtags = []
        hashtags = [str(h).strip() for h in hashtags if str(h).strip()][:4]

        while len(hashtags) < 4:
            hashtags.append("#instagram")

        hashtags = [h if h.startswith("#") else f"#{h}" for h in hashtags][:4]

        return {"caption": caption, "hashtags": hashtags}

    # -------------------- Identity helpers --------------------

    def compact_master_dna_for_poser(self, master_dna: str) -> str:
        s = (master_dna or "").strip()
        if len(s) <= 700:
            return s

        return (
            "LOCKED IDENTITY DNA (COMPACT) — DO NOT ALTER\n"
            "24–26-year-old South Asian (Indian-origin) female. Warm fair-to-medium golden-olive complexion; realistic skin texture with visible pores.\n"
            "Soft oval face, gently rounded cheeks, smooth feminine jawline, rounded chin.\n"
            "Medium-large almond deep-brown eyes with visible lid creases; dark medium-thick brows with soft natural arch.\n"
            "Straight proportionate nose with softly rounded refined tip.\n"
            "Naturally full balanced lips with defined cupid’s bow; muted rosy-pink tone.\n"
            "Dark brown to deep espresso hair, smooth to softly wavy, center/slightly off-center part with natural flyaways.\n"
            "Body structure: Hourglass (36-28-36). Identity must remain identical (no face drift/morphing/beautification)."
        )

    # -------------------- File helpers --------------------

    def _filelike_to_data_url(self, uploaded_file) -> str:
        content = uploaded_file.getvalue()
        mime = getattr(uploaded_file, "type", "image/jpeg") or "image/jpeg"
        b64 = base64.b64encode(content).decode("utf-8")
        return f"data:{mime};base64,{b64}"

    # -------------------- JSON utilities --------------------

    def _sanitize_json_text(self, s: str) -> str:
        if not s:
            return s
        # Basic cleanup: remove markdown code blocks if present
        if s.startswith("```json"):
            s = s[7:]
        if s.startswith("```"):
            s = s[3:]
        if s.endswith("```"):
            s = s[:-3]
        return s.strip()

    def _call_chat_json(self, messages: list, max_tokens: int = 1000) -> Dict[str, Any]:
        """
        Uses standard Chat Completions API with response_format={"type": "json_object"}.
        """
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
            text = resp.choices[0].message.content
            return json.loads(self._sanitize_json_text(text))
        except Exception as e:
            # Fallback for errors or model types that strictly don't support json_object
            print(f"JSON Error: {e}")
            return {}

    # -------------------- CLONER --------------------

    def cloner_analyze_filelike(self, uploaded_file, master_dna: str) -> Dict[str, Any]:
        data_url = self._filelike_to_data_url(uploaded_file)
        return self.cloner_analyze_data_url(data_url, master_dna)

    def cloner_analyze_data_url(self, data_url: str, master_dna: str) -> Dict[str, Any]:
        instructions = (
            "You are an expert prompt engineer.\n"
            "Analyze the image and produce a JSON prompt to recreate the scene.\n"
            "Return valid JSON with keys: pose, camera_angle, lighting_setup, background, hairstyle, makeup, attire, jewellery, body_structure, photo_realism_notes, negative_prompt, full_prompt.\n"
            "body_structure MUST be 'Hourglass (36-28-36)'.\n"
            "full_prompt MUST start with MASTER DNA verbatim."
        )

        user_text = (
            "MASTER DNA:\n"
            f"{master_dna}\n\n"
            "Analyze this image and produce the JSON."
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

        return self._call_chat_json(messages, max_tokens=1200)

    # -------------------- PROMPTER --------------------

    def prompter_build(self, master_dna: str, fields: Dict[str, str]) -> str:
        body_structure_line = "Body structure: Hourglass (36-28-36) — keep identical in every generation."
        parts = [
            master_dna.strip(),
            "",
            "PROMPT:",
            body_structure_line,
            f"Pose: {fields.get('pose', 'User-selected pose')}",
            f"Camera angle: {fields.get('camera_angle', '')}",
            f"Camera / lens / focus: {fields.get('camera_lens', '')}",
            f"Lighting setup: {fields.get('lighting', '')}",
            f"Background: {fields.get('background', '')}",
            f"Hairstyle: {fields.get('hairstyle', '')}",
            f"Makeup: {fields.get('makeup', '')}",
            f"Attire: {fields.get('attire', '')}",
            f"Jewellery: {fields.get('jewellery', '')}",
            "",
            "Quality + realism constraints:",
            "- shot on iPhone 17, f/16 look",
            "- realistic physics-based lighting and shadows",
            "- natural skin texture",
            "",
            "Negative prompt: blurry, bad anatomy, watermark, text",
        ]
        return "\n".join(parts)

    # -------------------- POSER --------------------

    def poser_variations_filelike(self, uploaded_file, master_dna: str, pose_style: str) -> Dict[str, Any]:
        data_url = self._filelike_to_data_url(uploaded_file)
        return self.poser_variations_data_url(data_url, master_dna, pose_style)

    def poser_variations_data_url(self, data_url: str, master_dna: str, pose_style: str) -> Dict[str, Any]:
        compact_dna = self.compact_master_dna_for_poser(master_dna)
        instructions = (
            "You are an expert prompt engineer.\n"
            "Create 5 different POSES based on the image style.\n"
            "Return JSON keys: scene_lock, pose_style, poses (array of objects with pose_name, pose_description, facial_expression)."
        )

        user_text = (
            f"Identity (Reference):\n{compact_dna}\n\n"
            f"Target Style: {pose_style}\n"
            "Analyze image and return JSON."
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

        data = self._call_chat_json(messages, max_tokens=1000)

        scene_lock = (data.get("scene_lock") or "").strip()
        poses = data.get("poses") or []
        
        prompts = []
        for p in poses[:5]:
            if isinstance(p, dict):
                prompts.append({
                    "pose_name": p.get("pose_name", ""),
                    "pose_description": p.get("pose_description", ""),
                    "facial_expression": p.get("facial_expression", ""),
                    "full_prompt": ""
                })

        return {
            "scene_lock": scene_lock,
            "pose_style": data.get("pose_style", pose_style),
            "prompts": prompts,
            "compact_master_dna": compact_dna,
        }

    # -------------------- PerfectCloner (STRICT SCHEMA) --------------------

    def _call_chat_schema(
        self,
        messages: list,
        schema_name: str,
        schema: Dict[str, Any],
        max_tokens: int = 1600,
    ) -> Dict[str, Any]:
        """
        Uses Structured Outputs (json_schema) via Chat Completions.
        """
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema_name,
                        "schema": schema,
                        "strict": True,
                    },
                },
            )
            return json.loads(resp.choices[0].message.content)
        except Exception as e:
            # Fallback if model doesn't support strict schema
            print(f"Schema Error: {e}. Falling back to standard JSON.")
            return self._call_chat_json(messages, max_tokens)

    def perfectcloner_analyze_filelike(
        self,
        uploaded_file,
        master_dna: str,
        identity_lock: bool = True,
    ) -> Dict[str, Any]:
        data_url = self._filelike_to_data_url(uploaded_file)
        return self.perfectcloner_analyze_data_url(data_url, master_dna, identity_lock=identity_lock)

    def perfectcloner_analyze_data_url(
        self,
        data_url: str,
        master_dna: str,
        identity_lock: bool = True,
    ) -> Dict[str, Any]:
        
        subject_placeholder = "[[SUBJECT:USER_FACE_AND_BODY]]"

        # Schema definition (same as before)
        schema = {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "version", "subject_placeholder", "identity_lock_enabled", "aspect_ratio",
                "camera", "lighting", "composition", "environment", "subject_appearance",
                "wardrobe", "hair_and_makeup", "props_and_objects", "postprocessing",
                "negative_prompt", "recreation_prompt", "insertion_instructions", "notes"
            ],
            "properties": {
                "version": {"type": "string"},
                "subject_placeholder": {"type": "string"},
                "identity_lock_enabled": {"type": "boolean"},
                "aspect_ratio": {"type": "string"},
                "camera": {
                    "type": "object", "additionalProperties": False,
                    "required": ["shot_type", "camera_angle", "framing", "lens", "focus_and_dof", "exposure_guess"],
                    "properties": {
                        "shot_type": {"type": "string"}, "camera_angle": {"type": "string"},
                        "framing": {"type": "string"},
                        "lens": {
                            "type": "object", "additionalProperties": False,
                            "required": ["lens_type", "focal_length_guess"],
                            "properties": {"lens_type": {"type": "string"}, "focal_length_guess": {"type": "string"}, "aperture_guess": {"type": "string"}}
                        },
                        "focus_and_dof": {
                            "type": "object", "additionalProperties": False,
                            "required": ["focus_target", "depth_of_field"],
                            "properties": {"focus_target": {"type": "string"}, "depth_of_field": {"type": "string"}}
                        },
                        "exposure_guess": {
                            "type": "object", "additionalProperties": False,
                            "required": ["iso_guess", "shutter_guess", "wb_guess"],
                            "properties": {"iso_guess": {"type": "string"}, "shutter_guess": {"type": "string"}, "wb_guess": {"type": "string"}}
                        }
                    }
                },
                "lighting": {
                    "type": "object", "additionalProperties": False,
                    "required": ["type", "direction", "softness", "contrast", "color_temperature", "shadow_character"],
                    "properties": {"type": {"type": "string"}, "direction": {"type": "string"}, "softness": {"type": "string"}, "contrast": {"type": "string"}, "color_temperature": {"type": "string"}, "shadow_character": {"type": "string"}}
                },
                "composition": {
                    "type": "object", "additionalProperties": False,
                    "required": ["pose", "expression", "eye_direction", "body_orientation", "background_depth", "scene_layout"],
                    "properties": {"pose": {"type": "string"}, "expression": {"type": "string"}, "eye_direction": {"type": "string"}, "body_orientation": {"type": "string"}, "background_depth": {"type": "string"}, "scene_layout": {"type": "string"}}
                },
                "environment": {
                    "type": "object", "additionalProperties": False,
                    "required": ["location_type", "background_details", "visible_text", "time_of_day"],
                    "properties": {"location_type": {"type": "string"}, "background_details": {"type": "string"}, "visible_text": {"type": "string"}, "time_of_day": {"type": "string"}}
                },
                "subject_appearance": {
                    "type": "object", "additionalProperties": False,
                    "required": ["age_range", "skin_tone", "body_structure", "other_physical_notes"],
                    "properties": {"age_range": {"type": "string"}, "skin_tone": {"type": "string"}, "body_structure": {"type": "string"}, "other_physical_notes": {"type": "string"}}
                },
                "wardrobe": {
                    "type": "object", "additionalProperties": False,
                    "required": ["outfit", "colors_materials", "fit", "accessories_jewellery"],
                    "properties": {"outfit": {"type": "string"}, "colors_materials": {"type": "string"}, "fit": {"type": "string"}, "accessories_jewellery": {"type": "string"}}
                },
                "hair_and_makeup": {
                    "type": "object", "additionalProperties": False,
                    "required": ["hair", "makeup"],
                    "properties": {"hair": {"type": "string"}, "makeup": {"type": "string"}}
                },
                "props_and_objects": {"type": "string"},
                "postprocessing": {
                    "type": "object", "additionalProperties": False,
                    "required": ["style", "color_grade", "sharpness", "grain", "retouching"],
                    "properties": {"style": {"type": "string"}, "color_grade": {"type": "string"}, "sharpness": {"type": "string"}, "grain": {"type": "string"}, "retouching": {"type": "string"}}
                },
                "negative_prompt": {"type": "string"},
                "recreation_prompt": {"type": "string"},
                "insertion_instructions": {"type": "string"},
                "notes": {"type": "string"}
            }
        }

        identity_header = (master_dna or "").strip() if identity_lock else ""
        identity_header_label = "MASTER DNA (identity lock)" if identity_lock else "MASTER DNA omitted"

        instructions = (
            "You are an expert prompt engineer.\n"
            "Analyze the reference image and output a SINGLE JSON object matching the Schema.\n"
            f"subject_placeholder MUST be exactly: {subject_placeholder}\n"
            "Build `recreation_prompt` starting with MASTER DNA (if locked) and using the placeholder."
        )

        user_text = (
            f"{identity_header_label}:\n{identity_header}\n\n"
            f"Subject placeholder: {subject_placeholder}\n"
            f"Lock enabled: {identity_lock}\n"
            "Analyze image and return JSON."
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

        return self._call_chat_schema(
            messages=messages,
            schema_name="perfectcloner_prompt_package",
            schema=schema,
            max_tokens=1600,
        )
