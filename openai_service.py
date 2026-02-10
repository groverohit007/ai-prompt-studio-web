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
           "Analyze the image and create 5 different POSES with matching facial expressions.\n\n"
           "You MUST keep: same identity, same face structure, same body, same attire, same makeup style, same background, same lighting, same camera style.\n"
           "You may change:\n"
           "- Body pose\n"
           "- Head angle\n"
           "- Facial expression\n"
           "- Eye direction\n"
           "- Hand placement\n\n"
           "Facial expressions must match the pose style and mood.\n"
          "For sensual or romantic styles, use soft, confident, emotionally expressive facial cues.\n\n"
          "Do NOT change identity, bone structure, or physical features.\n"
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
            "Naturally full balanced lips with defined cupid’s bow; muted rosy-pink tone; genuine smile with realistic teeth.\n"
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
        Try strict JSON mode if supported; fallback if not.
        """
        try:
            resp = self.client.responses.create(
                model=self.model,
                instructions=instructions,
                input=input_items,
                max_output_tokens=max_output_tokens,
                response_format={"type": "json_object"},
            )
            return self._extract_json_obj(resp.output_text)
        except TypeError:
            # response_format not supported in some older openai libs
            resp = self.client.responses.create(
                model=self.model,
                instructions=instructions + "\nReturn ONLY JSON. No markdown. No extra text.",
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
            "Analyze the image and create 5 different POSES while keeping everything else identical.\n\n"
            "You MUST keep: same identity, same face, same body, same attire, same makeup, same background, same lighting, same camera style.\n"
            "Only change: the pose.\n\n"
            "Return ONLY valid JSON with keys:\n"
            "scene_lock (string), pose_style (string), poses (array of exactly 5 objects).\n\n"
            "Each object must have:\n"
            "- pose_name (short)\n"
            "- pose_description (1–2 sentences, specific and visual)\n\n"
            "Rules:\n"
            "- DO NOT include long full prompts, only pose_name and pose_description.\n"
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

        # Normalize to the structure the UI expects: prompts=[{pose_name, pose_description, full_prompt}]
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
        "full_prompt": "",
    }
)

        return {
            "scene_lock": scene_lock,
            "pose_style": (data.get("pose_style") or pose_style).strip(),
            "prompts": prompts,
            "compact_master_dna": compact_dna,
        }
