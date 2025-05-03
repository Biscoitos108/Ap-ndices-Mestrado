import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Definir as dimensões da imagem de entrada
img_width, img_height = 180, 180

# Lista de caminhos para as imagens
img_paths = [
    r'D:/Users/Estevaos108/Desktop/Estevao Files/Imagens de Validacao - Orange2/Com Homoptera/Imagem-1.png',
    r'D:/Users/Estevaos108/Desktop/Estevao Files/Imagens de Validacao - Orange2/Com Homoptera/Imagem-2.png',
    r'D:/Users/Estevaos108/Desktop/Estevao Files/Imagens de Validacao - Orange2/Com Homoptera/Imagem-3.png',
    r'D:/Users/Estevaos108/Desktop/Estevao Files/Imagens de Validacao - Orange2/Com Homoptera/Imagem-4.png',
    r'D:/Users/Estevaos108/Desktop/Estevao Files/Imagens de Validacao - Orange2/Com Homoptera/Imagem-5.png',
    r'D:/Users/Estevaos108/Desktop/Estevao Files/Imagens de Validacao - Orange2/Sem Homoptera/Imagem-1.png',
    r'D:/Users/Estevaos108/Desktop/Estevao Files/Imagens de Validacao - Orange2/Sem Homoptera/Imagem-2.png',
    r'D:/Users/Estevaos108/Desktop/Estevao Files/Imagens de Validacao - Orange2/Sem Homoptera/Imagem-3.png',
    r'D:/Users/Estevaos108/Desktop/Estevao Files/Imagens de Validacao - Orange2/Sem Homoptera/Imagem-4.png',
    r'D:/Users/Estevaos108/Desktop/Estevao Files/Imagens de Validacao - Orange2/Sem Homoptera/Imagem-5.png'
]

# Definir as classes verdadeiras
y_true = [0] * 5 + [1] * 5  # 5 imagens "Com Homoptera" e 5 "Sem Homoptera"

# Carregar o modelo treinado
model_path = 'D:/Users/Estevaos108/Desktop/Estevao Files/modelo/best_model.keras'
if os.path.exists(model_path):
    model = tf.keras.models.load_model(model_path)
    print("Modelo carregado com sucesso!")
else:
    print(f"Erro: O modelo no caminho {model_path} não foi encontrado.")

# Listas para armazenar as previsões
y_pred_prob = []

# Carregar e processar as imagens e fazer previsões
for img_path in img_paths:
    if os.path.exists(img_path):
        img = load_img(img_path, target_size=(img_width, img_height))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Fazer a previsão
        predictions_saved_model = model.predict(img_array)
        y_pred_prob.append(predictions_saved_model[0][0])  # Adiciona a probabilidade da classe "Com Homoptera"

# Converter listas para numpy arrays
y_true = np.array(y_true)
y_pred_prob = np.array(y_pred_prob)

# Calcular a curva ROC e a área sob a curva (AUC)
fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
roc_auc = auc(fpr, tpr)

# Plotar a curva ROC
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='Curva ROC (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Curva Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

from sklearn.metrics import precision_recall_curve, f1_score

# Calcular precisão e recall para vários limiares
precision, recall, thresholds = precision_recall_curve(y_true, y_pred_prob)

# Calcular F1 para cada limiar
f1_scores = [f1_score(y_true, y_pred_prob >= t) for t in thresholds]

# Encontrar o limiar com o maior F1
best_threshold = thresholds[np.argmax(f1_scores)]
print(f"Limiar ideal para maximizar F1: {best_threshold}")

if os.path.exists(model_path):
    model = tf.keras.models.load_model(model_path)
    print("Modelo carregado com sucesso!")
else:
    print(f"Erro: O modelo no caminho {model_path} não foi encontrado.")

# Definir um dicionário para mapear índices de classe para nomes de classe
class_names = {0: 'Com Homoptera', 1: 'Sem Homoptera'}

# Configurar a visualização das imagens
plt.figure(figsize=(40, 30))  # Tamanho da figura

# Fazer a previsão com o novo limiar
for idx, img_path in enumerate(img_paths):
    if os.path.exists(img_path):
        img = load_img(img_path, target_size=(img_width, img_height))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Fazer a previsão
        predictions_saved_model = model.predict(img_array)
        predicted_prob = predictions_saved_model[0][0]

        # Usar o limiar ideal para classificar
        predicted_class_index = int(predicted_prob > best_threshold)
        predicted_class_name = class_names[predicted_class_index]

        confidence = predicted_prob if predicted_class_index == 0 else 1 - predicted_prob
        confidence_percentage = confidence * 100

        if confidence_percentage < 50:
            confidence_percentage = 100.00 - confidence_percentage

        plt.subplot(2, 5, idx + 1)
        plt.imshow(img)
        plt.title(f"{predicted_class_name} \n ({confidence_percentage:.2f}%)", size=8)
        plt.axis('off')
