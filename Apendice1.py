from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

image_generation = ImageDataGenerator(
        rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.1, zoom_range=0.1, horizontal_flip=True, fill_mode='nearest')
source_img = load_img('C:/Users/Estevao Cavalcante/Documents/Artigos Mestrado/Imagens de Validação/Sem Homoptera/2186_001.jpg')
x = img_to_array(source_img)
x = x.reshape((1,) + x.shape)
i = 0 
for batch in image_generation.flow(x, batch_size=1,
    save_to_dir='C:/Users/Estevao Cavalcante/Documents/Artigos Mestrado/Imagens de Validação/Sem Homoptera', 
    save_prefix='Nova_Imagem', save_format='jpg'):
    i += 1
    if i > 25:
       break
image_generation = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest')
source_img = load_img('C:/Users/Estevao Cavalcante/Documents/Artigos Mestrado/Imagens de Validação/Sem Homoptera/2186_001.jpg')
x = img_to_array(source_img)
x = x.reshape((1,) + x.shape)
i = 0
for batch in image_generation.flow(x, batch_size=1,
             save_to_dir='C:/Users/Estevao Cavalcante/Documents/Artigos Mestrado/Imagens de Validação/Sem Homoptera', 
            save_prefix='Nova_Imagem', save_format='jpg'):
        i += 1
        if i > 25:
           break
