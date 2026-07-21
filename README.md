# Katzen & Hunde Bildklassifizierung (PyTorch & CustomTkinter)

Ein Lehrprojekt zur binären Bildklassifizierung (Katzen vs. Hunde) mit einer eigenen CNN-Architektur in PyTorch und einer grafischen Benutzeroberfläche.

## Features
- **Eigenes CNN-Modell:** 4 Convolutional-Blöcke mit Batch Normalization, ReLU, MaxPool und Dropout zur Vermeidung von Overfitting
- **Binäre Klassifizierung:** Unterscheidung zwischen Katzen- und Hundebildern
- **GUI:** Benutzerfreundliche Desktop-Oberfläche mit CustomTkinter zur Bildauswahl und Anzeige der Vorhersage
- **Einfache Inferenz:** Bild-Preprocessing (Resizing auf 150x150 px, Normalisierung) und Sigmoid-Aktivierung für Wahrscheinlichkeitsberechnung

## Tech Stack
- Python 3.x
- PyTorch & Torchvision
- CustomTkinter
- Pillow (PIL)

## Hinweis
Dies ist ein **Schulungsprojekt**, das speziell für die Klassifizierung von Katzen und Hunden entwickelt wurde.
