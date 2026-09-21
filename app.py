# app.py
# เว็บแอปทำนายโอกาสรอดชีวิตผู้โดยสาร Titanic
# ห่อโมเดล Decision Tree ที่ฝึกและบันทึกไว้แล้ว (ไม่มีการเทรนใหม่ในไฟล์นี้)
# วิธีรัน: streamlit run app.py

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# ============================================================
# ค่าคงที่ -- แก้ตรงนี้จุดเดียวถ้าเปลี่ยนโมเดลหรือเกณฑ์
# ============================================================

MODEL_FILENAME = "titanic_model.joblib"

# ชื่อและลำดับฟีเจอร์ต้องตรงกับตอนเทรนเป๊ะ ๆ
# (คัดลอกจากผลของ inspect_model.py -> feature_names_in_)
FEATURE_ORDER = ["Pclass", "Sex_female", "Age", "Fare", "FamilySize"]

# คอลัมน์ที่ถูก MinMax สเกลเป็นช่วง 0-1 ตอนเทรน
# ต้องสเกลด้วยสูตร (x - min) / (max - min) เดิม ก่อนส่งเข้าโมเดล
SCALING_RANGES = {
    "Age": {"min": 0.42, "max": 80.0},
    "Fare": {"min": 0.0, "max": 512.3292},
}

# เกณฑ์ตัดสินใจ: ความน่าจะเป็นตั้งแต่ค่านี้ขึ้นไป ถือว่าทำนายเป็น 1
# *** บรรทัดนี้คือ "เกณฑ์เชิงธุรกิจ" ถ้าอยากเข้มงวดขึ้นให้เพิ่มค่า ***
DECISION_THRESHOLD = 0.5

LABEL_POSITIVE = "รอดชีวิต"
LABEL_NEGATIVE = "ไม่รอดชีวิต"
ADVICE_POSITIVE = "คาดว่ารอดชีวิต ลำดับความช่วยเหลือปกติ"
ADVICE_NEGATIVE = "คาดว่าไม่รอดชีวิต ควรจัดเป็นกลุ่มที่ต้องได้รับความช่วยเหลือก่อน"


# ============================================================
# ส่วนโหลดโมเดล
# ============================================================

@st.cache_resource
def load_model():
    """โหลดโมเดลจากไฟล์ .joblib

    @st.cache_resource ทำให้โหลดจากดิสก์แค่ครั้งเดียว
    แล้วเก็บไว้ในหน่วยความจำ ครั้งต่อ ๆ ไปที่ผู้ใช้กดปุ่ม
    Streamlit จะรันสคริปต์นี้ใหม่ทั้งไฟล์ แต่จะข้ามฟังก์ชันนี้ไป
    """
    # หาพาธจากตำแหน่งของ app.py ไม่ใช่โฟลเดอร์ที่สั่งรัน
    # เป็นพาธแบบสัมพัทธ์ จึงย้ายเครื่องหรือขึ้น cloud ได้โดยไม่ต้องแก้
    model_path = Path(__file__).parent / MODEL_FILENAME

    if not model_path.exists():
        return None, model_path

    return joblib.load(model_path), model_path


def scale_value(raw_value: float, column_name: str) -> float:
    """สเกลค่าดิบจากฟอร์มให้อยู่ช่วง 0-1 ด้วยสูตรเดียวกับตอนเทรน"""
    bounds = SCALING_RANGES[column_name]
    return (raw_value - bounds["min"]) / (bounds["max"] - bounds["min"])


def probability_to_color(probability: float) -> str:
    """แปลงความน่าจะเป็น (0-1) เป็นสี hex แบบไล่เฉด
    แดง (เสี่ยงสูง) -> เหลืองอำพัน (กึ่งกลาง) -> เขียว (ปลอดภัย)

    ใช้ 3 จุดสี (stop) แล้วไล่เฉดเชิงเส้นระหว่างจุดที่ใกล้ที่สุดสองจุด
    เพื่อให้สีเปลี่ยนแบบต่อเนื่องตามตัวเลขจริง ไม่ใช่แค่สลับ 2 สี
    """
    stops = [
        (0.0, (220, 38, 38)),    # แดง -- โอกาสรอดต่ำมาก
        (0.5, (245, 158, 11)),   # เหลืองอำพัน -- กึ่งกลาง ไม่ชัดเจน
        (1.0, (22, 163, 74)),    # เขียว -- โอกาสรอดสูงมาก
    ]

    for (pos_a, color_a), (pos_b, color_b) in zip(stops, stops[1:]):
        if pos_a <= probability <= pos_b:
            span = pos_b - pos_a
            ratio = 0.0 if span == 0 else (probability - pos_a) / span
            mixed = tuple(
                round(channel_a + (channel_b - channel_a) * ratio)
                for channel_a, channel_b in zip(color_a, color_b)
            )
            return "#{:02x}{:02x}{:02x}".format(*mixed)

    # ป้องกันกรณีค่าหลุดช่วง 0-1 เล็กน้อยจากการปัดเศษ
    return "#16a34a" if probability >= 1.0 else "#dc2626"


# ============================================================
# ตั้งค่าหน้าเว็บ
# ============================================================

st.set_page_config(
    page_title="ทำนายโอกาสรอดชีวิต Titanic",
    page_icon="🚢",
    layout="centered",
)

# CSS เสริมความสวยงาม -- ไม่เปลี่ยนโครงสร้างหรือตรรกะของแอป
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Kanit:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Kanit', sans-serif;
    }

    .block-container {
        max-width: 780px;
        padding-top: 1.2rem;
        padding-bottom: 3rem;
    }

    /* แถบ hero ด้านบน */
    .hero-band {
        background: linear-gradient(120deg, #0f172a 0%, #1e3a5f 55%, #0369a1 100%);
        border-radius: 18px;
        padding: 1.8rem 2rem;
        margin-bottom: 1.6rem;
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.25);
    }
    .hero-band h1 {
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 1.9rem !important;
        margin-bottom: 0.4rem !important;
    }
    .hero-band p {
        color: #cbd5e1 !important;
        font-size: 0.95rem !important;
        margin: 0 !important;
        line-height: 1.5;
    }

    /* การ์ดฟอร์มกรอกข้อมูล */
    .form-card {
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 1.4rem 1.6rem 0.6rem 1.6rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 4px 18px rgba(15, 23, 42, 0.05);
    }
    .form-card h3 {
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        margin-bottom: 0.6rem !important;
    }

    /* ปุ่มหลัก ทำนายผล */
    div.stButton > button[kind="primary"] {
        border-radius: 10px;
        height: 3.1rem;
        font-weight: 600;
        font-size: 1.05rem;
        background: linear-gradient(120deg, #0369a1, #0f172a);
        border: none;
        box-shadow: 0 6px 16px rgba(3, 105, 161, 0.35);
        transition: transform 0.15s ease;
    }
    div.stButton > button[kind="primary"]:hover {
        transform: translateY(-1px);
    }

    /* กล่อง expander */
    div[data-testid="stExpander"] {
        border-radius: 12px;
        border: 1px solid #e5e7eb;
        background-color: #ffffff;
    }

    /* แถบ sidebar */
    section[data-testid="stSidebar"] {
        background-color: #f8fafc;
    }

    hr {
        margin: 1.2rem 0 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero-band">
        <h1>🚢 ทำนายโอกาสรอดชีวิตผู้โดยสาร Titanic</h1>
        <p>กรอกข้อมูลผู้โดยสารด้านล่าง ระบบจะประเมินความน่าจะเป็นที่จะรอดชีวิต
        ด้วยโมเดล Decision Tree ที่ฝึกจากข้อมูลผู้โดยสารจริง สีของผลลัพธ์
        จะไล่เฉดตามระดับความน่าจะเป็น เพื่อให้ตีความได้รวดเร็วในหน้าเดียว</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# โหลดโมเดล ถ้าไม่เจอไฟล์ให้บอกให้ชัดแล้วหยุด ไม่ปล่อยให้พังทีหลัง
model, model_path = load_model()
if model is None:
    st.error(
        f"ไม่พบไฟล์โมเดล `{MODEL_FILENAME}`\n\n"
        f"ระบบมองหาที่: `{model_path}`\n\n"
        "วิธีแก้: วางไฟล์โมเดลไว้ในโฟลเดอร์เดียวกับ app.py "
        "และตรวจว่าชื่อไฟล์สะกดตรงกัน ไม่มีนามสกุลซ้อน เช่น .joblib.txt"
    )
    st.stop()

# ============================================================
# แถบด้านข้าง -- อธิบายแอปให้ผู้ใช้ที่ไม่รู้บริบทเข้าใจก่อนเริ่มกรอก
# ============================================================

with st.sidebar:
    st.markdown("### ℹ️ เกี่ยวกับแอปนี้")
    st.markdown(
        """
แอปนี้ทำนาย **โอกาสรอดชีวิต** ของผู้โดยสารเรือ Titanic
โดยใช้โมเดล **Decision Tree** (scikit-learn)

**ข้อมูลที่ใช้ฝึกโมเดล**
ชุดข้อมูลผู้โดยสารเรือ Titanic (Kaggle) จำนวน 5 ฟีเจอร์:
ชั้นโดยสาร, เพศ, อายุ, ค่าโดยสาร, ขนาดครอบครัว

**อ่านสีของผลลัพธ์**
- 🔴 แดง — ความน่าจะเป็นต่ำ
- 🟠 ส้ม/เหลือง — กึ่งกลาง ไม่ชัดเจน
- 🟢 เขียว — ความน่าจะเป็นสูง

**ข้อจำกัด**
- สะท้อนบริบทและค่านิยมของปี 1912 เท่านั้น
- ไม่ควรใช้ตัดสินสถานการณ์จริงในปัจจุบัน
- ใช้ตัวแปรเพียง 5 ตัว ยังไม่ครอบคลุมปัจจัยอื่น
        """
    )
    st.divider()
    st.caption(
        "⚠️ ผลจากโมเดลเป็นเพียง **ตัวช่วยประกอบการตัดสินใจ** "
        "คนยังเป็นผู้ตัดสินใจสุดท้ายเสมอ"
    )


# ============================================================
# ช่วงที่ 1 ของ pipeline: รับ input จากผู้ใช้
# ============================================================

st.markdown('<div class="form-card">', unsafe_allow_html=True)
st.markdown("### ข้อมูลผู้โดยสาร")

left_column, right_column = st.columns(2)

with left_column:
    pclass = st.selectbox(
        "ชั้นโดยสาร",
        options=[1, 2, 3],
        index=2,
        help="ระดับราคาตั๋ว 1 = ชั้นหนึ่ง (แพงที่สุด, ห้องอยู่ชั้นบน), "
             "2 = ชั้นสอง, 3 = ชั้นสาม (ถูกที่สุด, ห้องอยู่ชั้นล่างของเรือ)",
    )

    sex_label = st.radio(
        "เพศ",
        options=["หญิง", "ชาย"],
        index=1,
        horizontal=True,
        help="ตอนอพยพลงเรือชูชีพมีการให้สิทธิ์ผู้หญิงและเด็กก่อน "
             "เพศจึงเป็นปัจจัยที่มีผลมากในข้อมูลชุดนี้",
    )

    family_size = st.number_input(
        "ขนาดครอบครัวที่เดินทางด้วย (คน)",
        min_value=1,
        max_value=11,
        value=1,
        step=1,
        help="นับตัวผู้โดยสารเองรวมด้วย เช่น เดินทางคนเดียว = 1, "
             "มากับคู่สมรสและลูก 2 คน = 4",
    )

with right_column:
    age = st.slider(
        "อายุ (ปี)",
        min_value=0.0,
        max_value=80.0,
        value=30.0,
        step=1.0,
        help="อายุของผู้โดยสาร ข้อมูลที่ใช้ฝึกโมเดลมีช่วงตั้งแต่ "
             f"{SCALING_RANGES['Age']['min']} ถึง {SCALING_RANGES['Age']['max']} ปี",
    )

    fare = st.number_input(
        "ค่าโดยสาร (ปอนด์)",
        min_value=0.0,
        max_value=512.3292,
        value=15.0,
        step=1.0,
        help="ราคาตั๋วที่จ่ายจริง หน่วยเป็นเงินปอนด์ยุคปี 1912 "
             "ตั๋วชั้นสามมักต่ำกว่า 15 ส่วนชั้นหนึ่งมักสูงกว่า 50",
    )

st.markdown("</div>", unsafe_allow_html=True)

# แปลงตัวเลือกภาษาไทยกลับเป็น 0/1 ตามที่โมเดลเข้าใจ
# Sex_female: 1 = หญิง, 0 = ชาย
sex_female = 1 if sex_label == "หญิง" else 0

predict_clicked = st.button("ทำนายผล", type="primary", use_container_width=True)


# ============================================================
# ช่วงที่ 2 ของ pipeline: โมเดลประมวลผล
# ============================================================

if predict_clicked:
    # เก็บค่าดิบไว้แสดงในตารางตรวจสอบภายหลัง
    raw_values = {
        "Pclass": pclass,
        "Sex_female": sex_female,
        "Age": age,
        "Fare": fare,
        "FamilySize": family_size,
    }

    # สเกลเฉพาะคอลัมน์ที่เคยถูกสเกลตอนเทรน คอลัมน์อื่นใช้ค่าดิบ
    model_input_values = {
        column: scale_value(value, column) if column in SCALING_RANGES else value
        for column, value in raw_values.items()
    }

    # สร้าง DataFrame 1 แถว โดยบังคับลำดับคอลัมน์ตาม FEATURE_ORDER
    # ต้องเป็น DataFrame ที่ชื่อคอลัมน์ตรงกัน ไม่ใช่ list เปล่า ๆ
    # มิฉะนั้นจะเจอ error เรื่อง feature names mismatch
    input_frame = pd.DataFrame([model_input_values], columns=FEATURE_ORDER)

    # predict_proba คืนความน่าจะเป็นของทุกคลาสตามลำดับ model.classes_
    # คลาส 1 (รอดชีวิต) อยู่คอลัมน์ที่ index 1
    probabilities = model.predict_proba(input_frame)[0]
    survive_probability = float(probabilities[1])
    result_color = probability_to_color(survive_probability)

    # ============================================================
    # ช่วงที่ 3 ของ pipeline: แสดงผลลัพธ์
    # ============================================================

    st.markdown("#### 📋 ผลการทำนาย")

    # การ์ดผลลัพธ์แบบ gradient ที่ไล่สีตามความน่าจะเป็นจริง
    # ใช้สีเดียวกันทั้งพื้นหลังและแถบ progress เพื่อให้ตีความสอดคล้องกัน
    result_icon = "✅" if survive_probability >= DECISION_THRESHOLD else "⚠️"
    result_text = ADVICE_POSITIVE if survive_probability >= DECISION_THRESHOLD else ADVICE_NEGATIVE
    fill_percent = round(survive_probability * 100, 1)

    st.markdown(
        f"""
        <div style="
            background: linear-gradient(120deg, {result_color}22, {result_color}0d);
            border: 1px solid {result_color}55;
            border-radius: 18px;
            padding: 1.6rem 1.8rem;
            margin-bottom: 1rem;
        ">
            <div style="display:flex; justify-content:space-between; align-items:baseline; flex-wrap:wrap; gap:0.5rem;">
                <span style="font-size:0.95rem; color:#475569; font-weight:500;">
                    ความน่าจะเป็นที่จะ{LABEL_POSITIVE}
                </span>
                <span style="font-size:2.4rem; font-weight:700; color:{result_color};">
                    {survive_probability:.1%}
                </span>
            </div>
            <div style="
                width:100%; height:10px; border-radius:999px;
                background-color:#e2e8f0; margin-top:0.7rem; overflow:hidden;
            ">
                <div style="
                    width:{fill_percent}%; height:100%; border-radius:999px;
                    background-color:{result_color};
                "></div>
            </div>
            <div style="
                display:flex; justify-content:space-between;
                font-size:0.78rem; color:#94a3b8; margin-top:0.3rem;
            ">
                <span>{LABEL_NEGATIVE} 0%</span>
                <span>100% {LABEL_POSITIVE}</span>
            </div>
            <div style="
                margin-top:1.1rem; padding-top:1rem;
                border-top:1px solid {result_color}33;
                font-size:1rem; font-weight:600; color:#0f172a;
            ">
                {result_icon} {result_text}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("ดูวิธีคิดผลและข้อจำกัด"):
        st.markdown(
            f"""
**ระบบคิดผลอย่างไร**

1. รับค่าที่กรอกจากฟอร์ม
2. สเกลคอลัมน์ Age และ Fare ด้วยสูตร `(x - min) / (max - min)`
   ให้เป็นช่วง 0-1 เหมือนตอนฝึกโมเดล
3. เรียงค่าตามลำดับฟีเจอร์ที่โมเดลคาดหวัง แล้วส่งเข้า `predict_proba`
4. ถ้าความน่าจะเป็นตั้งแต่ **{DECISION_THRESHOLD:.2f}** ขึ้นไป
   ถือว่าทำนายเป็น "{LABEL_POSITIVE}" ต่ำกว่านั้นถือว่า "{LABEL_NEGATIVE}"
   เกณฑ์นี้เป็นค่าที่ทีมธุรกิจกำหนด ไม่ใช่ค่าที่โมเดลบังคับ

**สีของการ์ดผลลัพธ์มาจากไหน**

สีไล่เฉดต่อเนื่องจากแดง (ความน่าจะเป็น 0%) ผ่านเหลืองอำพัน (50%)
ไปจนถึงเขียว (100%) คำนวณจากตัวเลขความน่าจะเป็นจริงในฟังก์ชัน
`probability_to_color()` ไม่ใช่แค่สลับ 2 สีตามเกณฑ์ตัดสินใจ

**ข้อจำกัดที่ต้องรู้**

- โมเดลฝึกจากข้อมูลผู้โดยสาร Titanic ปี 1912 จึงสะท้อนบริบทและ
  ลำดับความช่วยเหลือของยุคนั้น ไม่ควรนำไปใช้ตัดสินสถานการณ์จริงในปัจจุบัน
- ใช้ข้อมูลเพียง 5 ตัวแปร ปัจจัยสำคัญอื่นอย่างตำแหน่งห้องพัก
  หรือสภาพร่างกาย ไม่ได้ถูกนำมาคิด
- ถ้ากรอกค่าที่อยู่นอกช่วงที่โมเดลเคยเห็นตอนฝึก ผลจะเชื่อถือได้น้อยลง

**ผลจากโมเดลเป็นเพียงตัวช่วยประกอบการตัดสินใจ คนเป็นผู้ตัดสินใจเสมอ**
            """
        )

        st.markdown("**ค่าที่กรอก เทียบกับ ค่าที่ส่งเข้าโมเดลจริง**")
        comparison_table = pd.DataFrame(
            {
                "ฟีเจอร์": FEATURE_ORDER,
                "ค่าที่กรอก": [raw_values[column] for column in FEATURE_ORDER],
                "ค่าที่ส่งเข้าโมเดล": [
                    round(model_input_values[column], 4) for column in FEATURE_ORDER
                ],
                "ถูกสเกลหรือไม่": [
                    "สเกลแล้ว" if column in SCALING_RANGES else "ใช้ค่าดิบ"
                    for column in FEATURE_ORDER
                ],
            }
        )
        st.dataframe(comparison_table, hide_index=True, use_container_width=True)