import streamlit as st
from ultralytics import YOLO
from PIL import Image
from pathlib import Path
import tempfile
import random
import cv2

st.set_page_config(page_title="Classification panneaux", layout="centered")
st.title("🚦 Classification des panneaux de signalisation")

MODEL_PATH = "runs/classify/train2/weights/best.pt"
DATASET_VAL = Path("cls_dataset/val")

model = YOLO(MODEL_PATH)

mode = st.radio(
    "Choisissez une source d’image",
    ["Uploader une image", "Webcam", "Image aléatoire du dataset"]
)

# -------- MODE 1 : UPLOAD --------
if mode == "Uploader une image":
    uploaded = st.file_uploader(
        "Glissez-déposez une image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded:
        image = Image.open(uploaded)
        st.image(image, caption="Image chargée", use_container_width=True)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            image.save(tmp.name)
            res = model(tmp.name)[0]

        cls = int(res.probs.top1)
        conf = float(res.probs.top1conf)
        nom = model.model.names[cls]

        st.success(f"Classe prédite : {nom}")
        st.write(f"Confiance : {conf:.2f}")

# -------- MODE 2 : DATASET RANDOM --------
elif mode == "Image aléatoire du dataset":
    if st.button("Choisir une image"):
        classe = random.choice([p for p in DATASET_VAL.iterdir() if p.is_dir()])
        img_path = random.choice(list(classe.iterdir()))

        image = Image.open(img_path)
        st.image(image, caption=f"Classe réelle : {classe.name}", use_container_width=True)

        res = model(str(img_path))[0]
        cls = int(res.probs.top1)
        conf = float(res.probs.top1conf)
        nom = model.model.names[cls]

        st.success(f"Classe prédite : {nom}")
        st.write(f"Confiance : {conf:.2f}")

# -------- MODE 3 : WEBCAM --------
elif mode == "Webcam":
    st.warning("Appuyez sur Q dans la fenêtre webcam pour quitter")

    run = st.button("Lancer la webcam")

    if run:
        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = model(rgb, imgsz=64)[0]

            cls = int(res.probs.top1)
            conf = float(res.probs.top1conf)
            nom = model.model.names[cls]

            cv2.putText(
                frame,
                f"{nom} ({conf:.2f})",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            cv2.imshow("Webcam - Classification panneaux", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
