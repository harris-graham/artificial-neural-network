import warnings
warnings.warn(
    "DEPRECATION WARNING: This Tkinter-based GUI is deprecated and will be removed in a future version. "
    "Please migrate to the new FastAPI web API. "
    "Start the API with: uvicorn src.api.main:app --reload "
    "and access the interactive documentation at http://localhost:8000/docs",
    DeprecationWarning,
    stacklevel=2
)

import numpy as np
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Fonction pour générer la séquence en fonction de la valeur de A
def generate_sequence(A):
    x = [0.1]
    for n in range(1, 500):
        x.append(A * x[n - 1] * (1 - x[n - 1]))
    return x

# Fonction pour exécuter le modèle avec la valeur de A et le nombre d'étapes sélectionnés
def run_model():
    selected_A = A_var.get()
    x = generate_sequence(selected_A)

    # Données d'entraînement
    X_train = np.array(x[:400])
    y_train = np.array(x[1:401])

    X_train = X_train.reshape(-1, 1, 1)
    y_train = y_train.reshape(-1, 1, 1)

    # Créer le modèle RNN/LSTM
    model = keras.Sequential()
    model.add(layers.LSTM(64, input_shape=(1, 1), return_sequences=True))
    model.add(layers.Dense(1))

    # Compiler le modèle
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Entraîner le modèle
    model.fit(X_train, y_train, epochs=100, batch_size=32)

    # Faire des prédictions
    predictions = model.predict(X_train.reshape(-1, 1, 1))

    # Récupérer le nombre d'étapes à l'avance sélectionné
    steps_ahead = steps_var.get()

    # Afficher les données dans la zone de texte
    text_widget.delete(1.0, tk.END)  # Effacer le contenu précédent
    for i in range(steps_ahead):
        text_widget.insert(tk.END, f"Étape {i+1}: {x[i]:.4f}\n")

    # Tracer le graphique
    ax.clear()
    ax.plot(X_train[:steps_ahead].flatten(), label='Valeurs existantes')
    ax.plot(predictions[:steps_ahead].flatten(), label='Prédictions')
    ax.legend()
    canvas.draw()

# Créer la fenêtre tkinter
root = tk.Tk()
root.title("Sélection de la valeur de A et du nombre d'étapes")

# Cadre en haut à gauche pour le choix de A
frame_A = ttk.Frame(root)
frame_A.grid(row=0, column=0, padx=10, pady=10, sticky="nw")

# Cadre au milieu à gauche pour le choix du nombre d'étapes
frame_steps = ttk.Frame(root)
frame_steps.grid(row=1, column=0, padx=10, pady=10, sticky="w")

# Cadre en bas à gauche pour l'affichage des données
frame_data = ttk.Frame(root)
frame_data.grid(row=2, column=0, padx=10, pady=10, sticky="sw")

# Cadre à droite pour le graphique
frame_graph = ttk.Frame(root)
frame_graph.grid(row=0, column=1, rowspan=3, padx=10, pady=10, sticky="nsew")

# Variable pour stocker la valeur de A sélectionnée
A_var = tk.IntVar()

# Créer des boutons radio pour choisir la valeur de A
ttk.Label(frame_A, text="Choisir la valeur de A:").grid(row=0, column=0, padx=10, pady=10, sticky="w")
ttk.Radiobutton(frame_A, text="A = 2", variable=A_var, value=2).grid(row=1, column=0, padx=10, pady=5, sticky="w")
ttk.Radiobutton(frame_A, text="A = 4", variable=A_var, value=4).grid(row=2, column=0, padx=10, pady=5, sticky="w")

# Variable pour stocker le nombre d'étapes à l'avance sélectionné
steps_var = tk.IntVar()
steps_var.set(3)  # Définir la valeur par défaut

# Créer des boutons radio pour choisir le nombre d'étapes à l'avance
ttk.Label(frame_steps, text="Nombre d'étapes à l'avance:").grid(row=0, column=0, padx=10, pady=10, sticky="w")
ttk.Radiobutton(frame_steps, text="3", variable=steps_var, value=3).grid(row=1, column=0, padx=10, pady=5, sticky="w")
ttk.Radiobutton(frame_steps, text="10", variable=steps_var, value=10).grid(row=2, column=0, padx=10, pady=5, sticky="w")
ttk.Radiobutton(frame_steps, text="20", variable=steps_var, value=20).grid(row=3, column=0, padx=10, pady=5, sticky="w")

# Bouton pour exécuter le modèle avec la valeur sélectionnée de A et le nombre d'étapes
ttk.Button(frame_A, text="Exécuter le modèle", command=run_model).grid(row=3, column=0, padx=10, pady=10, sticky="w")

# Zone de texte pour afficher les données
text_widget = tk.Text(frame_data, height=10, width=20)
text_widget.grid(row=0, column=0, padx=10, pady=10, sticky="w")

# Créer un graphique
fig, ax = plt.subplots()
ax.set_xlabel('Étapes')
ax.set_ylabel('Valeurs')
ax.set_title('Données et Prédictions')
canvas = FigureCanvasTkAgg(fig, master=frame_graph)
canvas.get_tk_widget().grid(row=0, column=0, padx=10, pady=10)

# Lancer l'interface utilisateur
root.mainloop()
