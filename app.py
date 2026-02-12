import os
import json
import base64
from PIL import Image, ImageFilter
import numpy as np
import streamlit as st
try:
    from streamlit_drawable_canvas import st_canvas
except Exception:
    st_canvas = None

from dotenv import load_dotenv

from openai_service import OpenAIService
from master_dna import DEFAULT_MASTER_DNA
import streamlit.components.v1 as components

def copy_button(label: str, text_to_copy: str):
    """Renders a real clipboard-copy button in the browser (Streamlit Cloud safe)."""
    import html
    import uuid

    safe = html.escape(text_to_copy or "")
    btn_id = f"copy_{uuid.uuid4().hex}"

    components.html(
        f"""
        <div style="margin: 6px 0;">
          <button id="{btn_id}"
            style="padding:8px 12px; border-radius:8px; border:1px solid #ccc; cursor:pointer;">
            {label}
          </button>
          <script>
            const btn = document.getElementById("{btn_id}");
            btn.addEventListener("click", async () => {{
              await navigator.clipboard.writeText("{safe}");
              btn.innerText = "✅ Copied!";
              setTimeout(() => btn.innerText = "{label}", 1200);
            }});
          </script>
        </div>
        """,
        height=60,
    )


load_dotenv()

st.set_page_config(page_title="AI Prompt Studio", layout="wide")
# --- Simple password gate (single-user) ---
APP_PASSWORD = os.getenv("APP_PASSWORD", "").strip()

if APP_PASSWORD:
    if "auth_ok" not in st.session_state:
        st.session_state.auth_ok = False

    if not st.session_state.auth_ok:
        st.title("AI Prompt Studio (Web)")
        pw = st.text_input("Enter password", type="password")

        if st.button("Login"):
            if pw == APP_PASSWORD:
                st.session_state.auth_ok = True
                st.success("Logged in!")
                st.rerun()
            else:
                st.error("Wrong password")

        st.stop()
# --- end password gate ---


API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
MODEL_DEFAULT = os.getenv("OPENAI_MODEL", "gpt-4.1-mini").strip()

if not API_KEY:
    st.error("Missing OPENAI_API_KEY. Add it to Streamlit Secrets or local .env file.")
    st.stop()

# Session state
if "master_prompt" not in st.session_state:
    st.session_state.master_prompt = DEFAULT_MASTER_DNA
if "model" not in st.session_state:
    st.session_state.model = MODEL_DEFAULT
if "poser_data" not in st.session_state:
    st.session_state.poser_data = None

st.title("AI Prompt Studio (Web)")

svc = OpenAIService(api_key=API_KEY, model=st.session_state.model)

# Sidebar
st.sidebar.header("App")
st.session_state.model = st.sidebar.text_input("Model", value=st.session_state.model)

tabs = st.tabs(["Cloner", "PerfectCloner", "Inpainting", "Prompter", "Poser", "Captions", "Settings"])

# ---------------- Cloner ----------------
with tabs[0]:
    st.subheader("Cloner")
    img = st.file_uploader("Upload a person image", type=["png", "jpg", "jpeg", "webp"])

    if img:
        st.image(img, caption="Uploaded image", use_container_width=True)

        if st.button("Analyze → Generate Prompt", key="cloner_btn"):
            with st.spinner("Analyzing image..."):
                data = svc.cloner_analyze_filelike(img, st.session_state.master_prompt)

            st.code(json.dumps(data, indent=2, ensure_ascii=False), language="json")
            st.text_area("Full Prompt", value=data.get("full_prompt", ""), height=280)


# ---------------- PerfectCloner ----------------
with tabs[1]:
    st.subheader("PerfectCloner (STRICT JSON Schema)")
    st.caption("Upload a reference image → get schema-validated prompt package (camera/lens/lighting/composition) with optional Master Identity Lock.")

    identity_lock_choice = st.radio(
        "Master Identity Lock",
        options=["ON (keep identity consistent)", "OFF (no identity lock)"],
        index=0,
        horizontal=True,
        key="pc_identity_lock",
    )
    identity_lock_on = identity_lock_choice.startswith("ON")

    pimg = st.file_uploader("Upload a reference image", type=["png", "jpg", "jpeg", "webp"], key="perfectcloner_upload")

    if pimg:
        st.image(pimg, caption="Reference image", use_container_width=True)

        if st.button("Analyze → Generate Perfect Prompt Package", key="perfectcloner_btn"):
            with st.spinner("Analyzing image (strict schema)..."):
                pkg = svc.perfectcloner_analyze_filelike(
                    pimg,
                    st.session_state.master_prompt,
                    identity_lock=identity_lock_on,
                )

            st.success("Done!")

            st.subheader("Prompt Package (JSON)")
            st.code(json.dumps(pkg, indent=2, ensure_ascii=False), language="json")

            recreation_prompt = (pkg.get("recreation_prompt") or "").strip()
            negative_prompt = (pkg.get("negative_prompt") or "").strip()
            placeholder = (pkg.get("subject_placeholder") or "").strip()

            st.subheader("Recreation Prompt")
            st.text_area("recreation_prompt", value=recreation_prompt, height=320, key="pc_recreation_prompt")
            copy_button("📋 Copy Recreation Prompt", recreation_prompt)

            st.subheader("Negative Prompt")
            st.text_area("negative_prompt", value=negative_prompt, height=140, key="pc_negative_prompt")
            copy_button("📋 Copy Negative Prompt", negative_prompt)

            st.subheader("Subject placeholder token")
            st.code(placeholder or "[[SUBJECT:USER_FACE_AND_BODY]]", language="text")

            st.subheader("Insertion instructions")
            st.info((pkg.get("insertion_instructions") or "").strip())

# ---------------- Inpainting ----------------
with tabs[2]:
    st.subheader("Inpainting (near-identical edits — template for Gemini Nano Banana Pro)")
    st.caption("This tab does NOT generate images. It builds a JSON request template you can send to Gemini later (reference image + mask + identity image + prompt).")

    base_img = st.file_uploader("Base image (the reference you want to keep)", type=["png", "jpg", "jpeg", "webp"], key="inp_base")
    mask_img = st.file_uploader("Mask image (white=edit, black=keep). Optional but recommended.", type=["png", "jpg", "jpeg", "webp"], key="inp_mask")
    id_img = st.file_uploader("Identity image (your face/body reference)", type=["png", "jpg", "jpeg", "webp"], key="inp_id")

    edit_prompt = st.text_area(
        "Edit prompt (describe what to change in the masked region; keep everything else identical)",
        value="Replace the person with [[SUBJECT:USER_FACE_AND_BODY]] while keeping pose, lighting, camera angle, background, and color grading identical.",
        height=160,
        key="inp_prompt",
    )


st.markdown("### Create mask helper (optional)")
st.caption("Draw white over the areas you want to edit (face/body). Everything else will be kept as-is. "
           "You can download the generated mask and upload it in the Mask uploader above.")

if base_img is None:
    st.info("Upload a base image above to enable the mask editor.")
else:
    if st_canvas is None:
        st.warning("Mask editor requires `streamlit-drawable-canvas`. Add it to requirements.txt: `streamlit-drawable-canvas`")
    else:
        # Load base image for background
        _base_pil = Image.open(base_img).convert("RGB")
        _w, _h = _base_pil.size

        
# Drawing mode controls
mode = st.radio(
    "Mask drawing mode",
    options=["Brush", "Rectangle", "Circle"],
    index=0,
    horizontal=True,
    key="mask_mode",
)
drawing_mode = {"Brush": "freedraw", "Rectangle": "rect", "Circle": "circle"}[mode]

brush_size = st.slider("Brush size / Stroke width", min_value=5, max_value=120, value=35, step=1, key="mask_brush")
stroke_width = brush_size

feather = st.slider("Feather / blur edges (px)", min_value=0, max_value=80, value=12, step=1, key="mask_feather")
mask_threshold = st.slider("Mask threshold", min_value=0, max_value=255, value=1, step=1, key="mask_threshold")

# Undo support: keep canvas objects list in session_state and rehydrate via initial_drawing
if "mask_initial_drawing" not in st.session_state:
    st.session_state["mask_initial_drawing"] = {"version": "4.4.0", "objects": []}

colu1, colu2, colu3 = st.columns([1, 1, 2])
with colu1:
    if st.button("↩️ Undo last", key="undo_mask_btn"):
        init = st.session_state.get("mask_initial_drawing") or {"version": "4.4.0", "objects": []}
        objs = init.get("objects") or []
        if len(objs) > 0:
            objs.pop()
            init["objects"] = objs
            st.session_state["mask_initial_drawing"] = init
        st.session_state.pop("inp_generated_mask_bytes", None)
        st.experimental_rerun()

with colu2:
    if st.button("🧽 Clear mask", key="clear_mask_btn"):
        st.session_state.pop("inp_generated_mask_bytes", None)
        st.session_state["mask_initial_drawing"] = {"version": "4.4.0", "objects": []}
        st.session_state.pop("mask_canvas", None)
        st.experimental_rerun()

# Canvas returns an RGBA image where painted pixels are in the stroke color
canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 1.0)",
    stroke_width=stroke_width,
    stroke_color="rgba(255, 255, 255, 1.0)",
    background_image=_base_pil,
    update_streamlit=True,
    height=_h,
    width=_w,
    drawing_mode=drawing_mode,
    initial_drawing=st.session_state.get("mask_initial_drawing"),
    key="mask_canvas",
)

if canvas_result.json_data is not None:
    try:
        st.session_state["mask_initial_drawing"] = canvas_result.json_data
    except Exception:
        pass

if canvas_result.image_data is not None:
    rgba = canvas_result.image_data.astype(np.uint8)
    alpha = rgba[:, :, 3]

    painted = (alpha > mask_threshold).astype(np.uint8) * 255
    mask_pil = Image.fromarray(painted, mode="L")

    if feather and feather > 0:
        mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(radius=float(feather)))

    st.image(mask_pil, caption="Generated mask (white=edit, black=keep)", use_container_width=True)

    import io
    buf = io.BytesIO()
    mask_pil.save(buf, format="PNG")
    st.session_state["inp_generated_mask_bytes"] = buf.getvalue()
    st.success("Generated mask is ready and will be used automatically (no need to upload).")

    st.download_button(
        "⬇️ Download mask.png",
        data=buf.getvalue(),
        file_name="mask.png",
        mime="image/png",
        key="download_mask_btn",
    )

    def _b64_any(upl_or_bytes):
        if upl_or_bytes is None:
            return None
        if isinstance(upl_or_bytes, (bytes, bytearray)):
            return base64.b64encode(upl_or_bytes).decode("utf-8")
        # Streamlit UploadedFile
        try:
            return base64.b64encode(upl_or_bytes.getvalue()).decode("utf-8")
        except Exception:
            return None

    if base_img:
        st.image(base_img, caption="Base image", use_container_width=True)

    if mask_img:
        st.image(mask_img, caption="Mask image", use_container_width=True)
    elif st.session_state.get("inp_generated_mask_bytes"):
        st.image(st.session_state["inp_generated_mask_bytes"], caption="Mask image (generated in-app)", use_container_width=True)

    if id_img:
        st.image(id_img, caption="Identity image", use_container_width=True)

    if st.button("Build Gemini Inpainting Request Template", key="inp_build"):
        payload = {
            "provider": "google_gemini",
            "model_suggestion": "gemini-3-pro-image-preview (Nano Banana Pro)",
            "notes": "Send this JSON to your Gemini image editing endpoint. Base + mask enable near-identical edits. Keep denoise/strength low for maximum similarity.",
            "inputs": {
                "base_image_b64": _b64_any(base_img),
                "mask_image_b64": _b64_any(mask_img) or _b64_any(st.session_state.get("inp_generated_mask_bytes")),
                "identity_image_b64": _b64_any(id_img),
                "prompt": edit_prompt.strip(),
            },
            "recommended_settings": {
                "preserve_background": True,
                "edit_strength_hint": "low",
                "guidance": "high",
                "aspect_ratio": "match_base_image",
            },
        }

        st.subheader("Gemini request template (JSON)")
        st.code(json.dumps(payload, indent=2, ensure_ascii=False), language="json")
        copy_button("📋 Copy Gemini JSON", json.dumps(payload, ensure_ascii=False))


# ---------------- Prompter ----------------
with tabs[5]:
    st.subheader("Prompter (editable fields)")

    defaults = {
        "pose": ["Confident standing", "Hand on hip", "Seated elegant pose", "Over-shoulder look"],
        "attire": ["Burgundy velvet Anarkali", "Saree (silk)", "Lehenga choli", "Casual jeans + top"],
        "makeup": ["Natural makeup + pink lipstick", "Soft glam", "Bold eyeliner look"],
        "camera_angle": ["Eye-level portrait", "Slight high angle", "Low angle (power pose)", "3/4 angle"],
        "camera_lens": ["iPhone 17, f/16 look, sharp focus", "85mm portrait look, shallow DOF", "35mm environmental portrait"],
        "lighting": ["Ring light front + soft ambient", "Window light side-lit", "Softbox key + fill", "Golden hour warm light"],
        "hairstyle": ["Long wavy center part", "High ponytail", "Loose curls", "Sleek straight hair"],
        "background": ["Indian living room", "Photo studio seamless", "Bedroom with Indian decor", "Modern cafe"],
        "jewellery": ["Silver hoop earrings", "Big jhumkas", "Minimal necklace", "Bangles on both wrists"],
    }

    col1, col2 = st.columns(2)
    with col1:
        pose = st.selectbox("Pose", defaults["pose"], index=0)
        attire = st.selectbox("Attire", defaults["attire"], index=0)
        makeup = st.selectbox("Makeup", defaults["makeup"], index=0)
        hairstyle = st.selectbox("Hairstyle", defaults["hairstyle"], index=0)

    with col2:
        camera_angle = st.selectbox("Camera angle", defaults["camera_angle"], index=0)
        camera_lens = st.selectbox("Camera / lens / focus", defaults["camera_lens"], index=0)
        lighting = st.selectbox("Lighting", defaults["lighting"], index=0)
        background = st.selectbox("Background", defaults["background"], index=0)
        jewellery = st.selectbox("Jewellery", defaults["jewellery"], index=0)

    extra_notes = st.text_input("Extra notes (optional)", "")

    if st.button("Generate Prompt", key="prompter_btn"):
        fields = {
            "pose": pose,
            "attire": attire,
            "makeup": makeup,
            "hairstyle": hairstyle,
            "camera_angle": camera_angle,
            "camera_lens": camera_lens,
            "lighting": lighting,
            "background": background,
            "jewellery": jewellery,
            "extra_notes": extra_notes,
        }
        prompt = svc.prompter_build(st.session_state.master_prompt, fields)
        st.text_area("Generated Prompt", value=prompt, height=380)

# ---------------- Poser ----------------
with tabs[6]:
    st.subheader("Poser (5 new pose prompts)")
    img2 = st.file_uploader("Upload AI model image", type=["png", "jpg", "jpeg", "webp"], key="poser_upload")
    pose_style = st.selectbox(
    "Pose style",
    ["Casual", "Elegant", "Sensual (tasteful)", "Romantic", "Confident", "Fitness", "Traditional", "Street style"],
    index=0
)

    if img2:
        st.image(img2, caption="Uploaded AI model image", use_container_width=True)

        if st.button("Generate 5 Pose Prompts", key="poser_btn"):
            with st.spinner("Creating pose variations..."):
                data = svc.poser_variations_filelike(img2, st.session_state.master_prompt, pose_style)
                st.session_state.poser_data = data
            st.success("Done!")

    data = st.session_state.poser_data
    if data and isinstance(data, dict):
        prompts = data.get("prompts", [])
        scene_lock = (data.get("scene_lock") or "").strip()
        compact_dna = (data.get("compact_master_dna") or "").strip()

        left, right = st.columns([1, 2])
        with left:
            names = [f"{i+1}. {p.get('pose_name','Pose')}" for i, p in enumerate(prompts)]
            choice = st.radio("Pose options", names, index=0)

        with right:
            idx = int(choice.split(".")[0]) - 1
            picked = prompts[idx] if 0 <= idx < len(prompts) else {}

            pose_name = (picked.get("pose_name") or "").strip()
            pose_desc = (picked.get("pose_description") or "").strip()
            face_expr = (picked.get("facial_expression") or "").strip()

            # Build the final usable prompt locally (prevents token/JSON break issues)
            full = "\n".join([
                compact_dna or st.session_state.master_prompt.strip(),
                "",
                "PROMPT:",
                "Body structure: Hourglass (36-28-36) — keep identical in every generation.",
                f"Pose: {pose_name}" if pose_name else "Pose: (selected pose)",
               f"Pose details: {pose_desc}" if pose_desc else "",
               f"Facial expression: {face_expr}" if face_expr else "",
                "",
                f"Scene lock (keep everything else identical): {scene_lock}" if scene_lock else "",
                "",
                "Quality + realism constraints:",
                "- shot on iPhone 17, f/16 look",
                "- realistic physics-based lighting and shadows",
                "- natural skin texture with pores and micro-details (not plastic, not overly smoothed)",
                "- sharp focus on subject, realistic depth of field",
                "",
                "Negative prompt:",
                "blurry, low-res, over-smoothed skin, plastic skin, uncanny face, deformed hands, extra fingers, bad anatomy, watermark, logo, text artifacts",
            ]).strip()

            st.text_area("Selected Prompt", value=full, height=420)

# ---------------- Captions ----------------
with tabs[3]:
    st.subheader("Captions (Instagram)")

    cap_img = st.file_uploader(
        "Upload a photo for caption",
        type=["png", "jpg", "jpeg", "webp"],
        key="caption_upload"
    )

    colA, colB = st.columns(2)
    with colA:
        style = st.selectbox(
            "Caption style",
            ["Engaging", "Funny", "Romantic", "Luxury", "Motivational", "Spiritual"],
            index=0
        )
    with colB:
        language = st.radio(
            "Language",
            ["English", "Hinglish", "Hindi"],
            horizontal=True
        )

    if cap_img:
        st.image(cap_img, caption="Uploaded photo", use_container_width=True)

        if st.button("Generate Caption", key="caption_btn"):
            with st.spinner("Writing caption..."):
                out = svc.captions_generate_filelike(cap_img, style=style, language=language)

            st.session_state.caption_out = out  # store result for copy buttons

    out = st.session_state.get("caption_out")
    if out:
        caption = out.get("caption", "")
        hashtags = out.get("hashtags", [])
        hashtags_text = " ".join(hashtags)

        st.text_area("Caption", value=caption, height=220, key="caption_text")
        copy_button("📋 Copy Caption", caption)

        st.text_input("Hashtags (4)", value=hashtags_text, key="hashtags_text")
        copy_button("📋 Copy Hashtags", hashtags_text)

# ---------------- Settings ----------------
with tabs[4]:
    st.subheader("Settings (Master DNA / Master Prompt)")
    st.session_state.master_prompt = st.text_area(
        "Master DNA / Master Prompt (used everywhere)",
        value=st.session_state.master_prompt,
        height=350
    )
    st.info("For single-user, this persists while your session is active. If you want permanent storage, we can save to a file.")
