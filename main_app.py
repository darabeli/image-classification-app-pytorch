import customtkinter as ctk
from PIL import Image
import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Importiert die Modellarchitektur und Hilfsfunktionen
from modelClass import CatsDogsCNN
from inference_logic import load_trained_model, run_prediction

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")


class ValidationWindow(ctk.CTkToplevel):
    """Separates Fenster für den Batch-Validierungstest."""

    def __init__(self, parent, model):
        super().__init__(parent)
        self.model = model

        self.title("Batch Validation Test")
        self.geometry("400x450")
        self.resizable(False, False)

        # Fokussiert das neue Fenster über dem Hauptfenster
        self.lift()
        self.attributes("-topmost", True)

        self.title_label = ctk.CTkLabel(
            self, text="Dataset Validation", font=ctk.CTkFont(size=18, weight="bold")
        )
        self.title_label.pack(pady=20)

        # Eingabefeld für die Anzahl der Bilder
        self.input_label = ctk.CTkLabel(
            self, text="Anzahl der Bilder für den Test eingeben:"
        )
        self.input_label.pack(pady=5)

        self.entry_count = ctk.CTkEntry(self, width=140, placeholder_text="z.B. 50")
        self.entry_count.pack(pady=10)

        self.btn_run_val = ctk.CTkButton(
            self, text="Test starten", command=self.run_batch_validation
        )
        self.btn_run_val.pack(pady=15)

        # Ergebnisanzeige
        self.results_frame = ctk.CTkFrame(self, width=320, height=180)
        self.results_frame.pack(pady=15)
        self.results_frame.pack_propagate(False)

        self.status_label = ctk.CTkLabel(
            self.results_frame, text="Status: Bereit", font=ctk.CTkFont(weight="bold")
        )
        self.status_label.pack(pady=15)

        self.res_accuracy = ctk.CTkLabel(self.results_frame, text="Accuracy: --")
        self.res_accuracy.pack(pady=5)

        self.res_loss = ctk.CTkLabel(self.results_frame, text="Validation Loss: --")
        self.res_loss.pack(pady=5)

    def run_batch_validation(self):
        """Validiert eine bestimmte Anzahl von Bildern aus dem Validation-Ordner."""
        input_val = self.entry_count.get().strip()
        if not input_val.isdigit():
            self.status_label.configure(text="Fehler: Ungültige Zahl!", text_color="red")
            return

        max_images = int(input_val)
        if max_images <= 0:
            self.status_label.configure(text="Fehler: Zahl muss > 0 sein", text_color="red")
            return

        self.status_label.configure(text="Berechne...", text_color="gray")
        self.update_idletasks()

        val_dir = "data/validation"
        if not os.path.exists(val_dir):
            self.status_label.configure(text="Fehler: 'data/validation' fehlt!", text_color="red")
            return

        # Transformationen für die Validierung
        transform = transforms.Compose([
            transforms.Resize((150, 150)),
            transforms.ToTensor()
        ])

        try:
            val_ds = datasets.ImageFolder(val_dir, transform=transform)
            # Nutzt DataLoader für ein effizientes Laden der Testdaten
            val_loader = DataLoader(val_ds, batch_size=32, shuffle=True)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(device)
            self.model.eval()

            loss_fn = torch.nn.BCEWithLogitsLoss()

            total_loss = 0.0
            correct = 0
            processed_images = 0

            with torch.no_grad():
                for images, labels in val_loader:
                    if processed_images >= max_images:
                        break

                    # Bestimmt die Größe des aktuellen Batches
                    current_batch_size = images.size(0)
                    if processed_images + current_batch_size > max_images:
                        # Schneidet den Batch ab, falls das Limit erreicht wird
                        needed = max_images - processed_images
                        images = images[:needed]
                        labels = labels[:needed]
                        current_batch_size = needed

                    images = images.to(device)
                    labels = labels.float().to(device).unsqueeze(1)

                    outputs = self.model(images)
                    loss = loss_fn(outputs, labels)

                    total_loss += loss.item() * current_batch_size
                    predictions = (torch.sigmoid(outputs) > 0.5)
                    correct += (predictions == labels).sum().item()

                    processed_images += current_batch_size

            # Berechnung der finalen Metriken
            final_loss = total_loss / processed_images
            final_acc = correct / processed_images

            self.status_label.configure(text="Test erfolgreich abgeschlossen!", text_color="green")
            self.res_accuracy.configure(text=f"Accuracy: {final_acc:.4f} ({correct}/{processed_images})")
            self.res_loss.configure(text=f"Validation Loss: {final_loss:.4f}")

        except Exception as e:
            self.status_label.configure(text=f"Fehler beim Laden!", text_color="red")
            print(f"Exception: {e}")


class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Cats vs Dogs Classifier")
        self.geometry("500x650")
        self.resizable(False, False)

        # Modell-Initialisierung beim Start
        self.raw_model = CatsDogsCNN()
        self.model = load_trained_model(self.raw_model, "cats_vs_dogs.pth")
        self.selected_image_path = None

        # UI-Widgets Hauptfenster
        self.title_label = ctk.CTkLabel(
            self, text="AI Image Classifier", font=ctk.CTkFont(size=24, weight="bold")
        )
        self.title_label.pack(pady=20)

        self.image_frame = ctk.CTkFrame(self, width=250, height=250)
        self.image_frame.pack(pady=10)
        self.image_frame.pack_propagate(False)

        self.no_image_label = ctk.CTkLabel(
            self.image_frame, text="No Image Selected", text_color="gray"
        )
        self.no_image_label.place(relx=0.5, rely=0.5, anchor="center")

        self.image_display = ctk.CTkLabel(self.image_frame, text="")

        self.btn_select = ctk.CTkButton(
            self, text="Select PNG Image", command=self.select_image, font=ctk.CTkFont(weight="bold")
        )
        self.btn_select.pack(pady=10)

        self.result_label = ctk.CTkLabel(
            self, text="Prediction: Waiting...", font=ctk.CTkFont(size=20, weight="bold")
        )
        self.result_label.pack(pady=10)

        self.prob_label = ctk.CTkLabel(self, text="Probability: --")
        self.prob_label.pack(pady=5)

        # Button zum Öffnen des zweiten Fensters
        self.btn_open_val = ctk.CTkButton(
            self,
            text="Open Validation Test",
            command=self.open_validation_window,
            fg_color="black",
            border_width=2
        )
        self.btn_open_val.pack(pady=25)

    def select_image(self):
        file_path = ctk.filedialog.askopenfilename(filetypes=[("PNG Images", "*.png")])
        if file_path:
            self.selected_image_path = file_path
            img = Image.open(file_path)
            ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(250, 250))

            self.no_image_label.pack_forget()
            self.image_display.configure(image=ctk_img)
            self.image_display.place(relx=0.5, rely=0.5, anchor="center")

            base_path, _ = os.path.splitext(file_path)
            prob = run_prediction(self.model, base_path)

            if prob is not None:
                self.prob_label.configure(text=f"Probability: {prob:.4f}")
                if prob > 0.5:
                    self.result_label.configure(text="Prediction: DOG 🐶", text_color="#1f538d")
                else:
                    self.result_label.configure(text="Prediction: CAT 🐱", text_color="#e67e22")

    def open_validation_window(self):
        """Öffnet das Subwindow für die Validierung."""
        ValidationWindow(self, self.model)


if __name__ == "__main__":
    app = App()
    app.mainloop()