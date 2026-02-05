import os
import json
import streamlit as st
from dotenv import load_dotenv

from openai_service import OpenAIService
from master_dna import DEFAULT_MASTER_DNA

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

tabs = st.tabs(["Cloner", "Prompter", "Poser", "Settings"])

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

# ---------------- Prompter ----------------
with tabs[1]:
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
with tabs[2]:
    st.subheader("Poser (5 new pose prompts)")
    img2 = st.file_uploader("Upload AI model image", type=["png", "jpg", "jpeg", "webp"], key="poser_upload")
    pose_style = st.selectbox("Pose style", ["Casual", "Elegant", "Sensual (tasteful)", "Fitness", "Traditional", "Street style"], index=0)

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

        left, right = st.columns([1, 2])
        with left:
            names = [f"{i+1}. {p.get('pose_name','Pose')}" for i, p in enumerate(prompts)]
            choice = st.radio("Pose options", names, index=0)

        with right:
            idx = int(choice.split(".")[0]) - 1
            picked = prompts[idx] if 0 <= idx < len(prompts) else {}

            full = (picked.get("full_prompt") or "").strip()
            if not full:
                # Fallback: build prompt if model forgot full_prompt
                master = st.session_state.master_prompt.strip()
                pose_name = (picked.get("pose_name") or "").strip()
                pose_desc = (picked.get("pose_description") or "").strip()
                full = "\n".join([
                    master,
                    "",
                    "PROMPT:",
                    "Body structure: Hourglass (36-28-36) — keep identical in every generation.",
                    "",
                    f"Pose: {pose_name}" if pose_name else "Pose: (selected pose)",
                    f"Pose details: {pose_desc}" if pose_desc else "",
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

# ---------------- Settings ----------------
with tabs[3]:
    st.subheader("Settings (Master DNA / Master Prompt)")
    st.session_state.master_prompt = st.text_area(
        "Master DNA / Master Prompt (used everywhere)",
        value=st.session_state.master_prompt,
        height=350
    )
    st.info("For single-user, this persists while your session is active. If you want permanent storage, we can save to a file.")
