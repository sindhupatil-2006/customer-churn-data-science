"""
Deep Learning Neural Network Pipeline Module
Author: Sindhu Patil (Data Science Intern)
"""

import os
os.environ['KERAS_BACKEND'] = 'torch'
import keras
from keras import layers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, average_precision_score
)

MODELS_DIR = os.path.join("outputs", "models")
RESULTS_DIR = os.path.join("outputs", "results")
FIGURES_DIR = os.path.join("outputs", "figures", "week5")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


def build_nn_architecture(input_dim):
    """
    Constructs a 4-layer Deep Feed-Forward Neural Network in Keras:
    - Input Layer: input_dim features
    - Dense(64, activation='relu') + Dropout(0.30)
    - Dense(32, activation='relu') + Dropout(0.20)
    - Dense(16, activation='relu')
    - Output Dense(1, activation='sigmoid')
    """
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation='relu', name='dense_1_64'),
        layers.Dropout(0.30, name='dropout_1_30'),
        layers.Dense(32, activation='relu', name='dense_2_32'),
        layers.Dropout(0.20, name='dropout_2_20'),
        layers.Dense(16, activation='relu', name='dense_3_16'),
        layers.Dense(1, activation='sigmoid', name='output_sigmoid')
    ], name="Churn_Deep_Neural_Network")
    
    return model


def compile_nn_model(model, learning_rate=0.001):
    """Compiles model with Adam optimizer, binary_crossentropy loss, and classification metrics."""
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc')
        ]
    )
    return model


def compute_train_class_weights(y_train):
    """Calculates balanced class weights for imbalanced training set."""
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    weights_dict = dict(zip(classes, weights))
    print("Computed Training Class Weights:", weights_dict)
    return weights_dict


def train_nn_model(model, X_train, y_train, X_val, y_val, class_weights, epochs=50, batch_size=32, patience=10):
    """Trains neural network with EarlyStopping callback monitoring val_loss."""
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=[early_stopping],
        verbose=1
    )

    model_path = os.path.join(MODELS_DIR, "deep_learning_nn_model.keras")
    model.save(model_path)
    print(f"Saved trained Keras Neural Network model to {model_path}")

    return model, history, model_path


def evaluate_nn_test_performance(model, X_test, y_test):
    """Evaluates final model on test set."""
    y_prob = model.predict(X_test).flatten()
    y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob)
    avg_prec = average_precision_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        'Model': 'Deep Learning Neural Network',
        'Accuracy': float(acc),
        'Precision': float(prec),
        'Recall': float(rec),
        'F1_Score': float(f1),
        'ROC_AUC': float(auc),
        'Average_Precision': float(avg_prec),
        'TN': int(cm[0, 0]),
        'FP': int(cm[0, 1]),
        'FN': int(cm[1, 0]),
        'TP': int(cm[1, 1])
    }

    return metrics, y_pred, y_prob, cm


def draw_architecture_diagram(input_dim, filename="neural_network_architecture.png"):
    """Generates a clean architecture diagram saved to outputs/figures/week5/."""
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.axis('off')

    layers_info = [
        (f"Input Layer\n({input_dim} Features)", "#3498db"),
        ("Dense Layer 1\n(64 Neurons, ReLU)", "#2ecc71"),
        ("Dropout Layer 1\n(Rate = 0.30)", "#e67e22"),
        ("Dense Layer 2\n(32 Neurons, ReLU)", "#2ecc71"),
        ("Dropout Layer 2\n(Rate = 0.20)", "#e67e22"),
        ("Dense Layer 3\n(16 Neurons, ReLU)", "#2ecc71"),
        ("Output Layer\n(1 Neuron, Sigmoid)", "#e74c3c")
    ]

    y_pos = np.linspace(0.9, 0.1, len(layers_info))

    for i, (text, color) in enumerate(layers_info):
        ax.text(0.5, y_pos[i], text, ha='center', va='center', fontsize=11, fontweight='bold',
                color='white', bbox=dict(boxstyle='round,pad=0.6', facecolor=color, edgecolor='none', alpha=0.9))
        if i < len(layers_info) - 1:
            ax.annotate('', xy=(0.5, y_pos[i+1] + 0.035), xytext=(0.5, y_pos[i] - 0.035),
                        arrowprops=dict(arrowstyle='->', color='black', lw=2))

    ax.set_title("Deep Learning Neural Network Architecture", fontsize=14, fontweight='bold', pad=20)
    
    filepath = os.path.join(FIGURES_DIR, filename)
    fig.tight_layout()
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved architecture diagram to {filepath}")
    return filepath
