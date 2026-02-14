import os
import json
import streamlit as st
from PIL import Image
import streamlit.components.v1 as components

# --- Check for Image Coordinates Library ---
try:
    from streamlit_image_coordinates import streamlit_image_coordinates
    HAS_COORDS = True
except ImportError:
    HAS_COORDS = False

from dotenv import load_dotenv
from openai_service import OpenAIService
from master_dna import DEFAULT_MASTER_DNA

# --- Copy Button Helper ---
def copy_button(label: str, text_to_copy: str):
    import html
    import uuid
    safe = html.escape(text_to_copy or "")
    btn_id = f"copy_{uuid.uuid4().hex}"
    components.html(
        f"""
        <div style="margin: 6px 0;">
          <button id="{btn_id}" style="padding:8px 12px; border-radius:8px; border:1px solid #ccc; cursor:pointer;">
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

# --- Password Gate ---
APP_PASSWORD = os.getenv("APP_PASSWORD", "").strip()
if APP_PASSWORD:
    if "auth_ok" not in st.session_state:
        st.session_state.auth_ok = False
    if not st.session_state.auth_ok:
        st.title("AI Prompt Studio")
        pw = st.text_input("Password", type="password")
        if st.button("Login"):
            if pw == APP_PASSWORD:
                st.session_state.auth_ok = True
                st.rerun()
            else:
                st.error("Wrong password")
        st.stop()

# --- Setup ---
API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
if not API_KEY:
    st.error("Missing OPENAI_API_KEY. Add it to Streamlit Secrets or local .env file.")
    st.stop()

if "master_prompt" not in st.session_state:
    st.session_state.master_prompt = DEFAULT_MASTER_DNA
if "model" not in st.session_state:
    st.session_state.model = "gpt-4o"
if "multi_angle_data" not in st.session_state:
    st.session_state.multi_angle_data = None
if "poser_data" not in st.session_state:
    st.session_state.poser_data = None

st.title("AI Prompt Studio (Web)")
svc = OpenAIService(api_key=API_KEY, model=st.session_state.model)

# --- Sidebar ---
st.sidebar.header("Configuration")
st.session_state.model = st.sidebar.text_input("OpenAI Model", value=st.session_state.model)

# --- TABS ---
# All tabs are now fully implemented
tabs = st.tabs(["Cloner", "PerfectCloner", "Multi-Angle Grid", "Digital Wardrobe", "Prompter", "Poser", "Captions", "Settings"])

# ---------------- Tab 0: Cloner ----------------
with tabs[0]:
    st.subheader("Cloner")
    img = st.file_uploader("Upload Person", type=["jpg", "png", "webp"], key="cloner_upl")
    if img and st.button("Analyze", key="cloner_btn"):
        with st.spinner("Analyzing..."):
            data = svc.cloner_analyze_filelike(img, st.session_state.master_prompt)
        st.code(json.dumps(data, indent=2))
        st.text_area("Full Prompt", value=data.get("full_prompt", ""), height=250)

# ---------------- Tab 1: PerfectCloner ----------------
with tabs[1]:
    st.subheader("PerfectCloner")
    pimg = st.file_uploader("Upload Reference", type=["jpg", "png", "webp"], key="pc_upl")
    identity_lock = st.checkbox("Enable Identity Lock", value=True)
    if pimg and st.button("Analyze Schema", key="pc_btn"):
        with st.spinner("Processing..."):
            data = svc.perfectcloner_analyze_filelike(pimg, st.session_state.master_prompt, identity_lock)
        
        st.success("Analysis Complete")
        rec_prompt = data.get("recreation_prompt", "")
        st.text_area("Recreation Prompt", value=rec_prompt, height=300)
        copy_button("📋 Copy Prompt", rec_prompt)
        st.json(data)

# ---------------- Tab 2: Multi-Angle Grid ----------------
with tabs[2]:
    st.subheader("Multi-Angle Pose Grid")
    st.info("1. Upload Reference -> 2. Plan 20 Angles -> 3. Upload Grid -> 4. Click to Prompt")
    
    ref_img = st.file_uploader("1. Upload Reference Character", type=["png", "jpg", "webp"], key="mag_ref")
    
    if ref_img:
        st.image(ref_img, width=200, caption="Reference")
        if st.button("Analyze & Plan 20 Angles", key="mag_plan_btn"):
            with st.spinner("Planning 20-angle character sheet..."):
                plan = svc.multi_angle_planner_filelike(ref_img, st.session_state.master_prompt)
                st.session_state.multi_angle_data = plan
                st.success("Plan generated!")
    
    plan_data = st.session_state.multi_angle_data
    if plan_data:
        st.divider()
        col1, col2 = st.columns([1, 1])
        with col1:
            grid_prompt = plan_data.get("grid_prompt", "")
            st.text_area("Grid Prompt (Use in DALL-E/Midjourney)", value=grid_prompt, height=150)
            copy_button("📋 Copy Grid Prompt", grid_prompt)
        with col2:
            st.markdown("**Angle List:**")
            with st.expander("View 20 Angles"):
                st.json(plan_data.get("angles", []))

        st.divider()
        st.markdown("### Select Angle")
        
        if not HAS_COORDS:
            st.warning("Install `streamlit-image-coordinates` to enable click-to-select.")
        else:
            grid_upload = st.file_uploader("Upload Generated Grid (4x5)", type=["png", "jpg"], key="mag_grid_upl")
            if grid_upload:
                value = streamlit_image_coordinates(grid_upload, key="grid_coords")
                if value:
                    pil_img = Image.open(grid_upload)
                    w, h = pil_img.size
                    col_idx = int(value["x"] // (w / 4))
                    row_idx = int(value["y"] // (h / 5))
                    angle_num = (row_idx * 4) + col_idx + 1
                    
                    angles = plan_data.get("angles", [])
                    if 0 < angle_num <= len(angles):
                        sel_angle = angles[angle_num - 1]
                        st.success(f"Selected Angle #{angle_num}: {sel_angle.get('name')}")
                        final_prompt = svc.build_physics_prompt(st.session_state.master_prompt, sel_angle)
                        st.text_area("Physics Prompt", value=final_prompt, height=250)
                        copy_button("📋 Copy Physics Prompt", final_prompt)

# ---------------- Tab 3: Digital Wardrobe (NEW) ----------------
with tabs[3]:
    st.subheader("Digital Wardrobe")
    st.info("Upload an image of an outfit (dress, suit, etc.). The AI will extract the 'Fashion DNA' and fuse it with your Master Identity.")

    wardrobe_img = st.file_uploader("Upload Outfit Reference", type=["png", "jpg", "webp"], key="wardrobe_upl")

    if wardrobe_img:
        st.image(wardrobe_img, width=250, caption="Outfit Reference")
        
        if st.button("🧵 Analyze & Wear", key="wardrobe_btn"):
            with st.spinner("Extracting textures and fabrics..."):
                w_data = svc.wardrobe_fuse_filelike(wardrobe_img, st.session_state.master_prompt)
            
            st.success("Outfit Fused!")
            
            st.markdown("### Outfit Analysis")
            st.write(w_data.get("outfit_description", "No description generated."))
            
            st.markdown("### Fused Prompt")
            fused_prompt = w_data.get("fused_prompt", "")
            st.text_area("Final Prompt", value=fused_prompt, height=300)
            copy_button("📋 Copy Fused Prompt", fused_prompt)

# ---------------- Tab 4: Prompter ----------------
with tabs[4]:
    st.subheader("Prompter")
    
    c1, c2 = st.columns(2)
    with c1:
        pose = st.selectbox("Pose", ["Confident", "Sitting", "Walking", "Close-up"], key="p_pose")
        attire = st.selectbox("Attire", ["Saree", "Business Suit", "Casual Jeans", "Evening Gown"], key="p_attire")
        lighting = st.selectbox("Lighting", ["Softbox", "Golden Hour", "Neon", "Natural"], key="p_light")
    with c2:
        cam = st.selectbox("Angle", ["Eye Level", "Low Angle", "Profile", "Top Down"], key="p_cam")
        bg = st.selectbox("Background", ["Living Room", "Street", "Studio", "Nature"], key="p_bg")
        jewel = st.selectbox("Jewellery", ["Minimal", "Heavy Gold", "Silver", "None"], key="p_jewel")

    if st.button("Generate Prompt", key="prompter_btn"):
        fields = {"pose": pose, "attire": attire, "lighting": lighting, "camera_angle": cam, "background": bg, "jewellery": jewel}
        prompt = svc.prompter_build(st.session_state.master_prompt, fields)
        st.text_area("Result", value=prompt, height=300)
        copy_button("📋 Copy", prompt)

# ---------------- Tab 5: Poser ----------------
with tabs[5]:
    st.subheader("Poser")
    poser_img = st.file_uploader("Upload Reference Pose", type=["jpg", "png"], key="poser_upl")
    style = st.selectbox("Style", ["Casual", "Elegant", "Edgy", "Professional"], key="poser_style")
    
    if poser_img and st.button("Generate Variations", key="poser_btn"):
        with st.spinner("Dreaming up poses..."):
            st.session_state.poser_data = svc.poser_variations_filelike(poser_img, st.session_state.master_prompt, style)
    
    if st.session_state.poser_data:
        data = st.session_state.poser_data
        prompts = data.get("prompts", [])
        
        # Display as Radio Selection
        names = [p.get("pose_name", "Pose") for p in prompts]
        selected_name = st.radio("Choose a Variation", names)
        
        # Find selected
        for p in prompts:
            if p.get("pose_name") == selected_name:
                full_text = f"{st.session_state.master_prompt}\n\nPOSE: {p.get('pose_name')}\nDETAILS: {p.get('pose_description')}\nEXPRESSION: {p.get('facial_expression')}\n\nSCENE LOCK: {data.get('scene_lock')}"
                st.text_area("Prompt", value=full_text, height=250)
                copy_button("📋 Copy Pose", full_text)

# ---------------- Tab 6: Captions ----------------
with tabs[6]:
    st.subheader("Captions")
    cap_img = st.file_uploader("Upload for Caption", type=["jpg", "png"], key="cap_upl")
    c_style = st.selectbox("Tone", ["Funny", "Serious", "Inspirational", "Sarcastic"], key="cap_style")
    c_lang = st.radio("Language", ["English", "Hindi", "Hinglish"], horizontal=True, key="cap_lang")
    
    if cap_img and st.button("Write Caption", key="cap_btn"):
        with st.spinner("Writing..."):
            res = svc.captions_generate_filelike(cap_img, c_style, c_lang)
        
        st.text_area("Caption", value=res.get("caption"), height=150)
        st.text_input("Hashtags", value=" ".join(res.get("hashtags", [])))

# ---------------- Tab 7: Settings ----------------
with tabs[7]:
    st.subheader("Settings")
    st.session_state.master_prompt = st.text_area("Master DNA", value=st.session_state.master_prompt, height=300)
