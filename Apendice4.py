import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
from PIL import Image

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

# Carregar o modelo treinado
model_path = 'D:/Users/Estevaos108/Desktop/Estevao Files/modelo/best_weights.h5'
if os.path.exists(model_path):
    model = tf.keras.models.load_model(model_path)
    print("Modelo carregado com sucesso!")
else:
    print(f"Erro: O modelo no caminho {model_path} não foi encontrado.")

# Definir um dicionário para mapear índices de classe para nomes de classe
class_names = {0: 'Com Homoptera', 1: 'Sem Homoptera'}

# Configurar a visualização das imagens
plt.figure(figsize=(40, 30))  # Tamanho da figura

# Carregar e processar as imagens e fazer previsões
for idx, img_path in enumerate(img_paths):
    if os.path.exists(img_path):
        img = load_img(img_path, target_size=(img_width, img_height))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Fazer a previsão
        predictions_saved_model = model.predict(img_array)
        predicted_class_index = int(predictions_saved_model[0] > 0.5)
        predicted_class_name = class_names[predicted_class_index]
        confidence = predictions_saved_model[0][0] if predicted_class_index == 0 else 1 - predictions_saved_model[0][0]
        confidence_percentage = confidence * 200
        
        if confidence_percentage < 50:
            confidence_percentage = 100.00 - confidence_percentage
        
        # Exibir imagem e previsão com a porcentagem
        plt.subplot(2, 5, idx + 1)  # 2 linhas, 5 colunas
        plt.imshow(img)
        plt.title(f"{predicted_class_name} \n ({confidence_percentage:.2f}%)", size = 8)
        plt.axis('off')
    else:
        print(f"Erro: O arquivo {img_path} não foi encontrado.")

# Salvar a imagem com resultados
plt.tight_layout()
plt.savefig('Mestrado_cigarrinhas_resultado.png')

# Exibir a imagem salva
img = Image.open('Mestrado_cigarrinhas_resultado.png')
img.show()
