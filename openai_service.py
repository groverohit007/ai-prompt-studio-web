import base64
import json
from typing import Any, Dict, List

from openai import OpenAI


class OpenAIService:
    """
    Streamlit-ready OpenAI service:
      - Supports Streamlit UploadedFile (file-like) via *_filelike methods
      - Robust JSON parsing & repair for model outputs
      - Enforces Body structure: Hourglass (36-28-36)
      - POSER returns *compact JSON* (no giant full_prompt strings) to avoid truncation/JSON issues.
        The UI builds the final full prompt locally.
      - Captions: returns caption + exactly 4 hashtags, supports English/Hinglish/Hindi.
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

        input_items = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            f"Caption style: {style}\n"
                            f"Language: {language}\n"
                            "Guidance:\n"
                            "- English: natural Instagram English.\n"
                            "- Hindi: Devanagari Hindi.\n"
                            "- Hinglish: Hindi + English mix in Roman script.\n"
                            "Return JSON only."
                        ),
                    },
                    {"type": "input_image", "image_url": data_url, "detail": "high"},
                ],
            }
        ]

        data = self._call_json(input_items, instructions, max_output_tokens=650)

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
        """
        Creates a short identity lock for Poser to avoid length/token issues.
        Keeps key facial traits + age/ethnicity + hair + skin + hourglass.
        If the provided DNA is already short, returns it unchanged.
        """
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

    # -------------------- File helpers (Streamlit) --------------------

    def _filelike_to_data_url(self, uploaded_file) -> str:
        """
        Streamlit UploadedFile -> data URL usable by Responses vision input.
        """
        content = uploaded_file.getvalue()
        mime = getattr(uploaded_file, "type", "image/jpeg") or "image/jpeg"
        b64 = base64.b64encode(content).decode("utf-8")
        return f"data:{mime};base64,{b64}"

    # -------------------- JSON utilities --------------------

    @staticmethod
    def _next_nonspace(s: str, i: int) -> str:
        n = len(s)
        while i < n and s[i] in " \t\r\n":
            i += 1
        return s[i] if i < n else ""

    def _sanitize_json_text(self, s: str) -> str:
        """
        Fixes common invalid JSON from LLM:
          - literal newlines/tabs inside quoted strings
          - unescaped internal quotes inside quoted strings
        """
        if not s:
            return s

        out = []
        in_str = False
        esc = False

        i = 0
        n = len(s)

        while i < n:
            ch = s[i]

            if in_str:
                if esc:
                    out.append(ch)
                    esc = False
                    i += 1
                    continue

                if ch == "\\":  # escape char
                    out.append(ch)
                    esc = True
                    i += 1
                    continue

                if ch == '"':
                    nxt = self._next_nonspace(s, i + 1)
                    # If next non-space isn't a JSON delimiter, treat as internal quote and escape it
                    if nxt and nxt not in [",", "}", "]"]:
                        out.append('\\"')
                        i += 1
                        continue

                    out.append(ch)
                    in_str = False
                    i += 1
                    continue

                if ch == "\n":
                    out.append("\\n")
                elif ch == "\r":
                    out.append("\\r")
                elif ch == "\t":
                    out.append("\\t")
                else:
                    out.append(ch)

                i += 1
                continue

            # not in string
            if ch == '"':
                out.append(ch)
                in_str = True
            else:
                out.append(ch)
            i += 1

        return "".join(out)

    def _repair_json_with_model(self, bad_text: str, max_output_tokens: int = 1200) -> Dict[str, Any]:
        fix = self.client.responses.create(
            model=self.model,
            instructions=(
                "You will be given text intended to be JSON but it may be invalid. "
                "Output STRICT VALID JSON only (no markdown, no extra text). "
                "Rules: escape all newlines as \\n and escape all internal quotes as \\\" inside strings."
            ),
            input=[{
                "role": "user",
                "content": [{"type": "input_text", "text": bad_text}],
            }],
            max_output_tokens=max_output_tokens,
        )
        fixed = (fix.output_text or "").strip()
        fixed = self._sanitize_json_text(fixed)
        return json.loads(fixed)

    def _extract_json_obj(self, text: str) -> Dict[str, Any]:
        raw = (text or "").strip()

        # 1) direct
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        # 2) sanitize
        cleaned = self._sanitize_json_text(raw)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # 3) extract {...}
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            maybe = cleaned[start:end + 1]
            try:
                return json.loads(maybe)
            except json.JSONDecodeError:
                return self._repair_json_with_model(maybe)

        return self._repair_json_with_model(cleaned)

    def _call_json(self, input_items: list, instructions: str, max_output_tokens: int = 900) -> Dict[str, Any]:
        """
        Request JSON output using Responses API structured text formatting.
        """
        # Newer OpenAI SDKs (Responses API) support structured output via text.format
        try:
            resp = self.client.responses.create(
                model=self.model,
                instructions=instructions,
                input=input_items,
                max_output_tokens=max_output_tokens,
                text={"format": {"type": "json_object"}},
            )
            return self._extract_json_obj(resp.output_text)
        except TypeError:
            # Fallback: no structured format available in this SDK version
            resp = self.client.responses.create(
                model=self.model,
                instructions=instructions + "
Return ONLY JSON. No markdown. No extra text.",
                input=input_items,
                max_output_tokens=max_output_tokens,
            )
            return self._extract_json_obj(resp.output_text)

    # -------------------- CLONER --------------------

    def cloner_analyze_filelike(self, uploaded_file, master_dna: str) -> Dict[str, Any]:
        data_url = self._filelike_to_data_url(uploaded_file)
        return self.cloner_analyze_data_url(data_url, master_dna)

    def cloner_analyze_data_url(self, data_url: str, master_dna: str) -> Dict[str, Any]:
        instructions = (
            "You are an expert prompt engineer for realistic portrait photography and AI image generation.\n"
            "Analyze the provided image and produce a prompt to recreate the SAME scene for a different AI model identity.\n\n"
            "Return ONLY valid JSON with these keys:\n"
            "pose, camera_angle, lighting_setup, background, hairstyle, makeup, attire, jewellery,\n"
            "body_structure, photo_realism_notes, negative_prompt, full_prompt.\n\n"
            "Rules:\n"
            "- Be specific, grounded in what you see (don’t invent props that are not visible).\n"
            "- Keep it safe-for-work.\n"
            "- body_structure MUST be exactly: 'Hourglass (36-28-36)'.\n"
            "- full_prompt MUST start with MASTER DNA verbatim.\n"
            "- full_prompt MUST include: 'Body structure: Hourglass (36-28-36)'.\n"
            "- IMPORTANT: Use \\n for newlines and escape quotes as \\\" in JSON strings.\n"
        )

        input_items = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "MASTER DNA (must be inserted verbatim at top of full_prompt):\n"
                            f"{master_dna}\n\n"
                            "Now analyze this image and produce the JSON."
                        ),
                    },
                    {"type": "input_image", "image_url": data_url, "detail": "high"},
                ],
            }
        ]

        return self._call_json(input_items, instructions, max_output_tokens=1200)

    # -------------------- PROMPTER (local build) --------------------

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
            "- natural skin texture with pores and micro-details (not plastic, not overly smoothed)",
            "- sharp focus on subject, realistic depth of field",
            "",
        ]

        extra = (fields.get("extra_notes") or "").strip()
        if extra:
            parts += ["Extra notes:", extra, ""]

        parts += [
            "Negative prompt:",
            "blurry, low-res, over-smoothed skin, plastic skin, uncanny face, deformed hands, extra fingers, bad anatomy, watermark, logo, text artifacts",
        ]
        return "\n".join(parts)

    # -------------------- POSER (compact JSON) --------------------

    def poser_variations_filelike(self, uploaded_file, master_dna: str, pose_style: str) -> Dict[str, Any]:
        data_url = self._filelike_to_data_url(uploaded_file)
        return self.poser_variations_data_url(data_url, master_dna, pose_style)

    def poser_variations_data_url(self, data_url: str, master_dna: str, pose_style: str) -> Dict[str, Any]:
        # Use compact identity DNA for poser to reduce token/length issues
        compact_dna = self.compact_master_dna_for_poser(master_dna)

        instructions = (
            "You are an expert prompt engineer.\n"
            "Analyze the image and create 5 different POSES with matching facial expressions.\n\n"
            "You MUST keep: same identity, same face structure, same body, same attire, same makeup style, "
            "same background, same lighting, same camera style.\n"
            "You may change: body pose, head angle, facial expression, eye direction, hand placement.\n\n"
            "Facial expressions must match the pose style and mood.\n"
            "For sensual/romantic styles: soft, confident, emotionally expressive facial cues (tasteful, safe-for-work).\n\n"
            "Return ONLY valid JSON with keys:\n"
            "scene_lock (string), pose_style (string), poses (array of exactly 5 objects).\n\n"
            "Each pose object must have:\n"
            "- pose_name (short)\n"
            "- pose_description (1–2 sentences, body position only)\n"
            "- facial_expression (short phrase describing emotion + eyes)\n\n"
            "Rules:\n"
            "- DO NOT include long full prompts.\n"
            "- Keep it tasteful and safe-for-work.\n"
            "- IMPORTANT: Use \\n for newlines and escape quotes as \\\" in JSON strings.\n"
        )

        input_items = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "MASTER DNA (for reference only; do not output it):\n"
                            f"{compact_dna}\n\n"
                            f"Pose style to target: {pose_style}\n\n"
                            "Now analyze the image and return the JSON."
                        ),
                    },
                    {"type": "input_image", "image_url": data_url, "detail": "high"},
                ],
            }
        ]

        data = self._call_json(input_items, instructions, max_output_tokens=900)

        # Normalize to the structure the UI expects: prompts=[{pose_name, pose_description, facial_expression, full_prompt}]
        scene_lock = (data.get("scene_lock") or "").strip()
        poses = data.get("poses") or data.get("prompts") or []
        if not isinstance(poses, list):
            poses = []

        prompts: List[Dict[str, str]] = []
        for p in poses[:5]:
            if not isinstance(p, dict):
                continue
            prompts.append(
                {
                    "pose_name": (p.get("pose_name") or "").strip(),
                    "pose_description": (p.get("pose_description") or "").strip(),
                    "facial_expression": (p.get("facial_expression") or "").strip(),
                    "full_prompt": "",  # UI will build this locally
                }
            )

        return {
            "scene_lock": scene_lock,
            "pose_style": (data.get("pose_style") or pose_style).strip(),
            "prompts": prompts,
            "compact_master_dna": compact_dna,
        }

# -------------------- PerfectCloner (STRICT JSON Schema) --------------------

def _call_json_schema(
    self,
    input_items: list,
    instructions: str,
    schema_name: str,
    schema: Dict[str, Any],
    max_output_tokens: int = 1600,
) -> Dict[str, Any]:
    """
    Strict JSON Schema output via Responses API (text.format).
    Falls back to json_object if json_schema is unavailable in your SDK/model.
    """
    try:
        resp = self.client.responses.create(
            model=self.model,
            instructions=instructions,
            input=input_items,
            max_output_tokens=max_output_tokens,
            text={
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "schema": schema,
                    "strict": True,
                }
            },
        )
        return self._extract_json_obj(resp.output_text)
    except TypeError:
        # Older SDKs: fall back to json_object
        resp = self.client.responses.create(
            model=self.model,
            instructions=instructions + "\nReturn ONLY JSON. No markdown. No extra text.",
            input=input_items,
            max_output_tokens=max_output_tokens,
            text={"format": {"type": "json_object"}},
        )
        return self._extract_json_obj(resp.output_text)

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
    """
    Analyze a reference image and return STRICT schema JSON containing:
      - camera + lens + lighting + composition
      - recreation_prompt (ready to use)
      - optional identity lock header (MASTER DNA) based on identity_lock toggle

    NOTE: This function does NOT generate images.
    """
    subject_placeholder = "[[SUBJECT:USER_FACE_AND_BODY]]"

    schema: Dict[str, Any] = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "version",
            "subject_placeholder",
            "identity_lock_enabled",
            "aspect_ratio",
            "camera",
            "lighting",
            "composition",
            "environment",
            "subject_appearance",
            "wardrobe",
            "hair_and_makeup",
            "props_and_objects",
            "postprocessing",
            "negative_prompt",
            "recreation_prompt",
            "insertion_instructions",
            "notes",
        ],
        "properties": {
            "version": {"type": "string"},
            "subject_placeholder": {"type": "string"},
            "identity_lock_enabled": {"type": "boolean"},
            "aspect_ratio": {"type": "string", "description": "e.g., 9:16, 16:9, 1:1"},
            "camera": {
                "type": "object",
                "additionalProperties": False,
                "required": ["shot_type", "camera_angle", "framing", "lens", "focus_and_dof", "exposure_guess"],
                "properties": {
                    "shot_type": {"type": "string"},
                    "camera_angle": {"type": "string"},
                    "framing": {"type": "string"},
                    "lens": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["lens_type", "focal_length_guess"],
                        "properties": {
                            "lens_type": {"type": "string"},
                            "focal_length_guess": {"type": "string"},
                            "aperture_guess": {"type": "string"},
                        },
                    },
                    "focus_and_dof": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["focus_target", "depth_of_field"],
                        "properties": {
                            "focus_target": {"type": "string"},
                            "depth_of_field": {"type": "string"},
                        },
                    },
                    "exposure_guess": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["iso_guess", "shutter_guess", "wb_guess"],
                        "properties": {
                            "iso_guess": {"type": "string"},
                            "shutter_guess": {"type": "string"},
                            "wb_guess": {"type": "string"},
                        },
                    },
                },
            },
            "lighting": {
                "type": "object",
                "additionalProperties": False,
                "required": ["type", "direction", "softness", "contrast", "color_temperature", "shadow_character"],
                "properties": {
                    "type": {"type": "string"},
                    "direction": {"type": "string"},
                    "softness": {"type": "string"},
                    "contrast": {"type": "string"},
                    "color_temperature": {"type": "string"},
                    "shadow_character": {"type": "string"},
                },
            },
            "composition": {
                "type": "object",
                "additionalProperties": False,
                "required": ["pose", "expression", "eye_direction", "body_orientation", "background_depth", "scene_layout"],
                "properties": {
                    "pose": {"type": "string"},
                    "expression": {"type": "string"},
                    "eye_direction": {"type": "string"},
                    "body_orientation": {"type": "string"},
                    "background_depth": {"type": "string"},
                    "scene_layout": {"type": "string"},
                },
            },
            "environment": {
                "type": "object",
                "additionalProperties": False,
                "required": ["location_type", "background_details", "visible_text", "time_of_day"],
                "properties": {
                    "location_type": {"type": "string"},
                    "background_details": {"type": "string"},
                    "visible_text": {"type": "string"},
                    "time_of_day": {"type": "string"},
                },
            },
            "subject_appearance": {
                "type": "object",
                "additionalProperties": False,
                "required": ["age_range", "skin_tone", "body_structure", "other_physical_notes"],
                "properties": {
                    "age_range": {"type": "string"},
                    "skin_tone": {"type": "string"},
                    "body_structure": {"type": "string"},
                    "other_physical_notes": {"type": "string"},
                },
            },
            "wardrobe": {
                "type": "object",
                "additionalProperties": False,
                "required": ["outfit", "colors_materials", "fit", "accessories_jewellery"],
                "properties": {
                    "outfit": {"type": "string"},
                    "colors_materials": {"type": "string"},
                    "fit": {"type": "string"},
                    "accessories_jewellery": {"type": "string"},
                },
            },
            "hair_and_makeup": {
                "type": "object",
                "additionalProperties": False,
                "required": ["hair", "makeup"],
                "properties": {
                    "hair": {"type": "string"},
                    "makeup": {"type": "string"},
                },
            },
            "props_and_objects": {"type": "string"},
            "postprocessing": {
                "type": "object",
                "additionalProperties": False,
                "required": ["style", "color_grade", "sharpness", "grain", "retouching"],
                "properties": {
                    "style": {"type": "string"},
                    "color_grade": {"type": "string"},
                    "sharpness": {"type": "string"},
                    "grain": {"type": "string"},
                    "retouching": {"type": "string"},
                },
            },
            "negative_prompt": {"type": "string"},
            "recreation_prompt": {"type": "string"},
            "insertion_instructions": {"type": "string"},
            "notes": {"type": "string"},
        },
    }

    identity_header = (master_dna or "").strip() if identity_lock else ""
    identity_header_label = "MASTER DNA (identity lock)" if identity_lock else "MASTER DNA omitted (no identity lock)"

    instructions = (
        "You are an expert prompt engineer for photorealistic image recreation.\n"
        "Analyze the reference image and output a SINGLE JSON object matching the provided JSON Schema.\n"
        "Rules:\n"
        f"- subject_placeholder MUST be exactly: {subject_placeholder}\n"
        "- Keep it safe-for-work.\n"
        "- Do not invent objects not visible.\n"
        "- Ensure the returned JSON conforms to the schema exactly.\n\n"
        "CRITICAL: Build `recreation_prompt` as a ready-to-use generation prompt that:\n"
        + ("- starts with MASTER DNA verbatim\n" if identity_lock else "- does NOT include MASTER DNA or any identity-lock text\n")
        + f"- uses {subject_placeholder} as the ONLY subject identity reference\n"
        "- includes camera + lens + lighting + composition in natural language\n"
        "- preserves the reference scene as closely as possible\n"
    )

    input_items = [
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": (
                        f"{identity_header_label}:\n{identity_header}\n\n"
                        f"Subject placeholder token (must appear in JSON and in recreation_prompt): {subject_placeholder}\n"
                        f"identity_lock_enabled must be: {str(identity_lock).lower()}\n"
                        "Now analyze this reference image and return STRICT JSON per schema.\n"
                        "If MASTER DNA is omitted, do NOT add any identity-lock phrasing."
                    ),
                },
                {"type": "input_image", "image_url": data_url, "detail": "high"},
            ],
        }
    ]

    data = self._call_json_schema(
        input_items=input_items,
        instructions=instructions,
        schema_name="perfectcloner_prompt_package",
        schema=schema,
        max_output_tokens=1600,
    )

    return data
