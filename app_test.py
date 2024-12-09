import cv2
import threading
import time
from playsound import playsound
from ultralytics import YOLO

ALERT_FILE = "./sound.mp3"
ALERT_REPEATS = 5
DATA_PATH = "./dataSet/data.yaml"
MODEL_PATH = "./runs/detect/train/weights/best.pt"

def train_model(data_path: str, epochs: int) -> None:
    try:
        model = YOLO("./yolov8m.pt")
        model.train(data=data_path, epochs=epochs)
        print("Entraînement terminé.")
    except Exception as e:
        print(f"Erreur lors de l'entraînement : {e}")

def validate_model(model_path: str, data_path: str) -> None:
    try:
        model = YOLO(model_path)
        validation_results = model.val(data=data_path)
        print("Validation terminée. Métriques :", validation_results)
    except Exception as e:
        print(f"Erreur lors de la validation : {e}")

def predict(model, frame, conf: float):
    try:
        results = model.predict(source=frame, show=False, conf=conf, stream=False)
        return results
    except Exception as e:
        print(f"Erreur lors de la prédiction : {e}")
        return None

def play_alert(alert_file: str) -> None:
    for _ in range(ALERT_REPEATS):
        try:
            print("!!! WARNING: FEU DÉTECTÉ !!!")
            playsound(alert_file)
        except Exception as e:
            print(f"Erreur lors de la lecture de l'alerte sonore : {e}")
            break

def show_blinking_text(frame, text="FIRE FIRE", x=50, y=50, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, color=(0, 0, 255), thickness=2, blink_rate=0.5):
    """Affiche un texte clignotant sur la frame."""
    blink_interval = blink_rate
    current_time = time.time()
    if int(current_time / blink_interval) % 2 == 0:  # Clignote toutes les 'blink_rate' secondes
        cv2.putText(frame, text, (x, y), font, font_scale, color, thickness)

def show_results(model, frame, conf: float, alert_played: bool) -> bool:
    print("start detection ")
    results = predict(model, frame, conf)
    if results:
        for result in results:
            for box in result.boxes:
                class_index = int(box.cls)
                class_name = model.names[class_index]
                print(f"Classe détectée : {class_name}")

                # Dessiner une boîte autour du feu détecté
                if class_name == "fire" and not alert_played:
                    print("Feu détecté, alerte en cours.")
                    alert_thread = threading.Thread(target=play_alert, args=(ALERT_FILE,))
                    alert_thread.daemon = True
                    alert_thread.start()
                    alert_played = True
    return alert_played

def show(model, source=0, conf=0.5):
    cap = cv2.VideoCapture(source)  
    if not cap.isOpened():
        print("Erreur d'ouverture de la webcam.")
        return

    alert_played = False
    frame_count = 0
    prediction_interval = 30  # Effectuer une prédiction tous les 30 frames

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erreur de lecture de l'image")
            break

        frame_count += 1
        # Effectuer une prédiction tous les prediction_interval frames
        if frame_count % prediction_interval == 0:
            alert_played = show_results(model, frame, conf, alert_played)

        # Affichage du texte clignotant
        if alert_played:
            show_blinking_text(frame, "FIRE FIRE", 50, 50, blink_rate=0.5)

        # Affichage du frame
        cv2.imshow("Webcam", frame)

        # Appuyer sur 'q' pour quitter
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    model = YOLO(MODEL_PATH)
    show(model=model, source=0, conf=0.7)

if __name__ == "__main__":
    main()
