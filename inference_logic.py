# inference_logic.py
import torch
from torchvision import transforms
from PIL import Image

IMG_SIZE = 150


def load_trained_model(model_architecture, weights_path="cats_vs_dogs.pth"):
    """Lädt die Gewichte einmalig in die Modellstruktur."""
    model_architecture.load_state_dict(
        torch.load(weights_path, map_location="cpu")
    )
    model_architecture.eval()
    return model_architecture


def run_prediction(model, image_name):
    """Führt die eigentliche Vorhersage für ein Bild aus."""
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor()
    ])

    image_path = f"{image_name}.png"

    try:
        img = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"Fehler: Die Datei '{image_path}' wurde nicht gefunden.")
        return None

    # Bild-Preprozessierung
    x = transform(img).unsqueeze(0)

    device = next(model.parameters()).device
    x = x.to(device)

    with torch.no_grad():
        output = model(x)
        probability = torch.sigmoid(output).item()

    return probability