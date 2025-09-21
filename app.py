import io
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn import datasets, metrics, model_selection
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(
    page_title="AI Starter (Streamlit + scikit-learn)",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI Starter: Classification ด้วย Streamlit + scikit-learn")
st.caption("ตัวอย่างโปรเจกต์ AI เบื้องต้น: โหลดข้อมูล → แบ่ง train/test → เทรนโมเดล → ประเมินผล → พยากรณ์ข้อมูลใหม่")

# ------------------------------
# Utilities
# ------------------------------
@st.cache_data
def load_iris_df():
    iris = datasets.load_iris(as_frame=True)
    df = iris.frame.copy()
    df.rename(columns={
        "sepal length (cm)": "sepal_length",
        "sepal width (cm)": "sepal_width",
        "petal length (cm)": "petal_length",
        "petal width (cm)": "petal_width",
        "target": "species"
    }, inplace=True)
    target_names = iris.target_names
    return df, target_names

def build_model(model_name, params):
    """สร้าง Pipeline มาตรฐาน: StandardScaler (ยกเว้น RF) + Model"""
    if model_name == "Logistic Regression":
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=params.get("C", 1.0),
                max_iter=1000,
                random_state=42
            ))
        ])
    elif model_name == "KNN":
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(
                n_neighbors=params.get("n_neighbors", 5),
                weights=params.get("weights", "uniform")
            ))
        ])
    elif model_name == "SVM (RBF)":
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(
                C=params.get("C", 1.0),
                gamma=params.get("gamma", "scale"),
                probability=True,
                random_state=42
            ))
        ])
    elif model_name == "Random Forest":
        # RF ไม่จำเป็นต้องสเกล
        pipe = Pipeline([
            ("clf", RandomForestClassifier(
                n_estimators=params.get("n_estimators", 200),
                max_depth=params.get("max_depth", None),
                random_state=42
            ))
        ])
    else:
        raise ValueError("ไม่รู้จักโมเดล")
    return pipe

def plot_confusion_matrix(cm, classes):
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    im = ax.imshow(cm, interpolation='nearest')
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        ylabel='True label',
        xlabel='Predicted label',
        title='Confusion Matrix'
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return fig

# ------------------------------
# Sidebar Controls
# ------------------------------
st.sidebar.header("⚙️ ตั้งค่าโมเดล & การเทรน")

model_name = st.sidebar.selectbox(
    "เลือกโมเดล",
    ["Logistic Regression", "KNN", "SVM (RBF)", "Random Forest"]
)

test_size = st.sidebar.slider("สัดส่วน Test size", 0.1, 0.5, 0.2, 0.05)
shuffle_data = st.sidebar.checkbox("สลับข้อมูล (Shuffle)", True)
random_state = st.sidebar.number_input("Random state", value=42, step=1)

st.sidebar.subheader("Hyperparameters")
params = {}
if model_name == "Logistic Regression":
    params["C"] = st.sidebar.number_input("C (regularization)", min_value=0.01, value=1.0, step=0.1)
elif model_name == "KNN":
    params["n_neighbors"] = st.sidebar.slider("n_neighbors (k)", 1, 20, 5, 1)
    params["weights"] = st.sidebar.selectbox("weights", ["uniform", "distance"])
elif model_name == "SVM (RBF)":
    params["C"] = st.sidebar.number_input("C (regularization)", min_value=0.01, value=1.0, step=0.1)
    params["gamma"] = st.sidebar.selectbox("gamma", ["scale", "auto"])
elif model_name == "Random Forest":
    params["n_estimators"] = st.sidebar.slider("n_estimators", 50, 500, 200, 50)
    max_depth_opt = st.sidebar.selectbox("max_depth", ["None", 3, 5, 8, 12])
    params["max_depth"] = None if max_depth_opt == "None" else int(max_depth_opt)

col_left, col_right = st.columns([1.1, 1])

# ------------------------------
# Load data
# ------------------------------
df, target_names = load_iris_df()
feature_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
X = df[feature_cols].values
y = df["species"].values

with col_left:
    st.subheader("👀 ดูข้อมูลตัวอย่าง")
    st.dataframe(df.sample(10, random_state=42), use_container_width=True)
    st.write(f"จำนวนแถวทั้งหมด: **{len(df)}**")

# ------------------------------
# Train/Test split
# ------------------------------
X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X, y, test_size=test_size, shuffle=shuffle_data, random_state=random_state, stratify=y
)

# ------------------------------
# Train Model
# ------------------------------
with col_right:
    st.subheader("🧠 เทรนโมเดล")
    train_btn = st.button("Train / Evaluate", type="primary", use_container_width=True)

    if train_btn:
        model = build_model(model_name, params)
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        acc = metrics.accuracy_score(y_test, y_pred)
        cm = metrics.confusion_matrix(y_test, y_pred)

        st.success(f"Accuracy (Test): **{acc:.4f}**")
        fig_cm = plot_confusion_matrix(cm, target_names)
        st.pyplot(fig_cm, use_container_width=True)

        # Classification report (ย่อให้สวย)
        report = metrics.classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
        rep_df = pd.DataFrame(report).transpose()
        st.markdown("**Classification Report**")
        st.dataframe(rep_df.round(3), use_container_width=True)

        # Save model to bytes for download
        buffer = io.BytesIO()
        joblib.dump(model, buffer)
        buffer.seek(0)
        st.download_button(
            "💾 ดาวน์โหลดโมเดลที่เทรนแล้ว (.joblib)",
            data=buffer,
            file_name=f"iris_{model_name.replace(' ', '_').lower()}_model.joblib",
            mime="application/octet-stream",
            use_container_width=True
        )

# ------------------------------
# Predict on user data
# ------------------------------
st.markdown("---")
st.subheader("🔮 พยากรณ์จากไฟล์ที่คุณอัปโหลด (CSV)")

st.markdown(
    """
- เตรียมไฟล์ CSV ที่มีคอลัมน์: `sepal_length, sepal_width, petal_length, petal_width`  
- โมเดลจะถูกเทรนใหม่ด้วยพารามิเตอร์ที่ตั้งไว้ด้านซ้ายก่อนพยากรณ์ (เพื่อความง่ายในการเดโม่)
"""
)

uploaded = st.file_uploader("อัปโหลดไฟล์ CSV", type=["csv"])
if uploaded is not None:
    try:
        user_df = pd.read_csv(uploaded)
        missing = [c for c in feature_cols if c not in user_df.columns]
        if missing:
            st.error(f"คอลัมน์หายไป: {missing}")
        else:
            st.write("ตัวอย่างข้อมูลที่อัปโหลด:")
            st.dataframe(user_df.head(), use_container_width=True)

            # เทรนโมเดลใหม่อย่างรวดเร็ว (เหมือนด้านบน)
            model_for_pred = build_model(model_name, params)
            model_for_pred.fit(X, y)

            preds = model_for_pred.predict(user_df[feature_cols].values)
            pred_names = [target_names[i] if isinstance(i, (int, np.integer)) else target_names[int(i)] for i in preds]
            out = user_df.copy()
            out["predicted_species"] = pred_names

            st.success("พยากรณ์เสร็จสิ้น ✅")
            st.dataframe(out.head(20), use_container_width=True)

            # สำหรับดาวน์โหลดผลลัพธ์
            csv_bytes = out.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ ดาวน์โหลดผลพยากรณ์ (CSV)",
                data=csv_bytes,
                file_name="predictions.csv",
                mime="text/csv"
            )
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการอ่านไฟล์/พยากรณ์: {e}")

# ------------------------------
# How it works
# ------------------------------
with st.expander("ℹ️ อธิบายขั้นตอนทำงาน (เหมาะสำหรับผู้เริ่มต้น)"):
    st.markdown("""
1) **โหลดข้อมูล Iris**: มี 4 ฟีเจอร์ (sepal_length, sepal_width, petal_length, petal_width) และคลาส 3 ชนิดของดอกไม้  
2) **แบ่ง Train/Test**: ใช้ `train_test_split` เพื่อประเมินโมเดลบนข้อมูลที่ไม่เคยเห็น  
3) **สร้างโมเดล**: เลือกโมเดลและตั้งค่า Hyperparameters ใน Sidebar  
   - Logistic Regression / KNN / SVM / Random Forest  
   - ใช้ `Pipeline` เพื่อรวม `StandardScaler` กับตัวจำแนก (ยกเว้น RF)  
4) **เทรนและประเมิน**: แสดง Accuracy, Confusion Matrix และ Classification Report  
5) **พยากรณ์ข้อมูลใหม่**: อัปโหลด CSV ที่มีคอลัมน์ 4 ตัว แล้วได้ผลคลาสที่คาดการณ์  
""")
