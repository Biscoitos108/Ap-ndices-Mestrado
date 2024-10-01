import cv2
import os

def process_image(input_path, output_path, size=(224, 224), format='png'):
    image = cv2.imread(input_path)
    if image is None:
        print(f"Erro ao ler a imagem {input_path}")
        return

    # Redimensionar a imagem
    resized_image = cv2.resize(image, size)

    # Salvar a imagem no formato desejado
    filename = os.path.splitext(os.path.basename(input_path))[0] + '.' + format
    cv2.imwrite(os.path.join(output_path, filename), resized_image)

# Diretórios de entrada e saída
input_dir = 'D:/Users/Estevaos108/Desktop/Estevao Files/Imagens de Validacao - Orange/Sem Homoptera'
output_dir = 'D:/Users/Estevaos108/Desktop/Estevao Files/Imagens de Validacao - Orange/Sem Homoptera'

# Criar diretório de saída se não existir
os.makedirs(output_dir, exist_ok=True)

# Processar todas as imagens no diretório de entrada
for filename in os.listdir(input_dir):
    if filename.endswith(('.png', '.jpg', '.jpeg')):
        process_image(os.path.join(input_dir, filename), output_dir)
