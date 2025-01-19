
import gradio as gr
import cv2
from fer import FER

# Función para detectar emociones
def detectar_emociones(imagen):
    detector = FER(mtcnn=True)
    emociones_detectadas = detector.detect_emotions(imagen)
    resultados = []

    for emocion in emociones_detectadas:
        emocion_dominante = max(emocion["emotions"], key=emocion["emotions"].get)
        resultados.append(emocion_dominante)

    if not resultados:
        return "No se detectaron emociones en la imagen."
    return "Emociones detectadas: " + ", ".join(resultados)

# Interfaz Gradio
interfaz = gr.Interface(
    fn=detectar_emociones,
    inputs=gr.Image(type="numpy", label="Sube una imagen"),
    outputs=gr.Textbox(label="Resultado"),
    title="Detector de Emociones",
    description="Sube una imagen para detectar emociones humanas en español.",
)

# Ejecutar la aplicación
interfaz.launch(server_name="0.0.0.0", server_port=8080)
