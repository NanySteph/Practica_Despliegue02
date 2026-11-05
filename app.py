import csv
import random
import pandas as pd
import streamlit as st
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(page_title="Billboard Top 1 Predictor", page_icon="🎵", layout="centered")

st.markdown("""
<style>
.stApp { background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%); color: #f5f5f5; }
label, .stMarkdown p { color: #f5f5f5 !important; }
.hero { text-align:center; padding: 1.8rem 1rem; border-radius: 20px;
        background: linear-gradient(90deg, #ff006e, #8338ec, #3a86ff);
        box-shadow: 0 10px 30px rgba(0,0,0,.4); margin-bottom: 1.5rem; }
.hero h1 { color: white; font-size: 2.3rem; margin: 0; }
.hero p { color: #f0f0f0; margin-top: .4rem; }
.result-yes { background: linear-gradient(90deg,#06d6a0,#118ab2); padding:1.4rem;
              border-radius:16px; text-align:center; font-size:1.5rem; font-weight:800; color:white; }
.result-no { background: linear-gradient(90deg,#ef476f,#8338ec); padding:1.4rem;
             border-radius:16px; text-align:center; font-size:1.5rem; font-weight:800; color:white; }
.firma { position: fixed; bottom: 12px; right: 18px; z-index: 999;
         background: rgba(0,0,0,.45); color: #ffd166; padding: 6px 14px;
         border-radius: 12px; font-size: .85rem; font-weight: 600;
         border: 1px solid rgba(255,209,102,.4); }
</style>
<div class="firma">✨ Stephanie Mercedes Machado Alvarado</div>
""", unsafe_allow_html=True)

# ─────────── Datos y modelo ───────────
@st.cache_resource
def entrenar():
    df = pd.read_csv("billboard.csv")
    malas = df["title"].isna()
    if malas.any():
        rep = df.loc[malas, "id"].apply(lambda x: next(csv.reader([str(x)])))
        rep = pd.DataFrame(rep.tolist(), columns=df.columns)
        df = df.astype(object)
        df.loc[malas, df.columns] = rep.values
    for col in ["durationSeg", "top"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["top", "durationSeg", "artist_type", "tempo", "genre"])
    df = df[df["durationSeg"] > 0]          # quitar duraciones inválidas
    df["top"] = df["top"].astype(int)

    X = pd.get_dummies(df[["artist_type", "tempo", "genre", "durationSeg"]])
    y = df["top"]
    # class_weight="balanced" compensa que haya muchas más canciones "No Top 1"
    modelo = DecisionTreeClassifier(max_depth=4, min_samples_leaf=10,
                                    class_weight="balanced", random_state=30)
    modelo.fit(X, y)
    return df, X.columns, modelo


def fuegos_artificiales():
    colores = ["#ffd166", "#ff006e", "#06d6a0", "#3a86ff", "#8338ec", "#ffffff"]
    explosiones = ""
    for i in range(7):
        color = random.choice(colores)
        chispas = ", ".join(
            f"{random.randint(-160, 160)}px {random.randint(-160, 160)}px 0 2px {color}"
            for _ in range(40)
        )
        x = random.randint(10, 90)
        y = random.randint(10, 55)
        retraso = round(i * 0.4, 1)
        explosiones += (
            f'<div class="fw" style="left:{x}%; top:{y}%; '
            f'animation-delay:{retraso}s; box-shadow:{chispas};"></div>'
        )
    return f"""
<style>
.fw-capa {{ position: fixed; inset: 0; pointer-events: none; z-index: 9999;
            animation: desaparecer 5s forwards; }}
.fw {{ position: absolute; width: 4px; height: 4px; border-radius: 50%;
       opacity: 0; transform: scale(0); animation: explotar 1.6s ease-out 3; }}
@keyframes explotar {{
  0%   {{ transform: scale(0);   opacity: 1; }}
  70%  {{ transform: scale(1);   opacity: 1; }}
  100% {{ transform: scale(1.2); opacity: 0; }}
}}
.fw-texto {{ position: absolute; top: 40%; width: 100%; text-align: center;
             font-size: 4.5rem; font-weight: 900; color: #ffd166;
             text-shadow: 0 0 20px #ff006e, 0 0 40px #8338ec;
             animation: latir 0.8s ease-in-out infinite alternate; }}
@keyframes latir {{ from {{ transform: scale(1); }} to {{ transform: scale(1.15); }} }}
@keyframes desaparecer {{ 0%, 85% {{ opacity: 1; }} 100% {{ opacity: 0; }} }}
</style>
<div class="fw-capa">{explosiones}<div class="fw-texto">🏆 ¡TOP 1! 🏆</div></div>
"""


try:
    df, columnas, modelo = entrenar()
except FileNotFoundError:
    st.error("No se encontró **billboard.csv** en el repositorio.")
    st.stop()

# ─────────── Interfaz ───────────
st.markdown("""
<div class="hero">
  <h1>🎤 ¿Llegará al #1 de Billboard? 🏆</h1>
  <p>Crea tu canción y el Árbol de Decisión te dirá su destino</p>
</div>
""", unsafe_allow_html=True)

a, b = st.columns(2)
with a:
    tipo = st.selectbox("👩‍🎤 Tipo de artista", sorted(df["artist_type"].unique()))
    tempo = st.selectbox("🥁 Tempo", sorted(df["tempo"].unique()))
with b:
    genero = st.selectbox("🎸 Género", sorted(df["genre"].unique()))
    dur = st.slider("⏱️ Duración (segundos)", 60, 600, 210)
    st.caption(f"{dur // 60}:{dur % 60:02d} minutos")

if st.button("🔮 ¡Predecir!", use_container_width=True):
    nueva = pd.get_dummies(pd.DataFrame([{"artist_type": tipo, "tempo": tempo,
                                          "genre": genero, "durationSeg": dur}]))
    nueva = nueva.reindex(columns=columnas, fill_value=False)
    prob = modelo.predict_proba(nueva)[0][1]

    if prob >= 0.5:
        st.markdown(fuegos_artificiales(), unsafe_allow_html=True)
        st.markdown('<div class="result-yes">🏆 ¡Esta canción llegaría al TOP 1!</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<div class="result-no">🎵 Esta canción no llegaría al Top 1</div>',
                    unsafe_allow_html=True)
    st.write("")
    st.progress(float(prob), text=f"Probabilidad de ser #1: {prob:.1%}")
