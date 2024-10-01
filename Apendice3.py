import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import GlobalAveragePooling2D
import matplotlib.pyplot as plt
from PIL import Image
import os

# Configurações de semente para reprodutibilidade
tf.compat.v1.set_random_seed(2019)

# Função para verificar e ajustar o tamanho das imagens
def resize_images(directory, target_size=(180, 180)):
    for subdir, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(subdir, file)
            try:
                with Image.open(file_path) as img:
                    if img.size != target_size:
                        img = img.resize(target_size)
                        img.save(file_path)
            except Exception as e:
                print(f"Erro ao processar a imagem {file_path}: {e}")

# Verificando e ajustando imagens de treino e validação
resize_images('D:/Users/Estevaos108/Desktop/Estevao Files/Imagens de Treino')
resize_images('D:/Users/Estevaos108/Desktop/Estevao Files/Imagens de Validacao')

# rodar se já tiver modelo pré-treinado:
loaded_model = tf.keras.models.load_model('D:/Users/Estevaos108/Desktop/Estevao Files/modelo/best_weights.h5')

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(180,180,3)),
    BatchNormalization(),
    Conv2D(32, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.5),
    Conv2D(64, (3,3), activation='relu'),
    BatchNormalization(),
    Conv2D(64, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.5),
    Conv2D(128, (3,3), activation='relu'),
    BatchNormalization(),
    GlobalAveragePooling2D(),  # Substitui o Flatten
    Dense(100, activation='relu', kernel_regularizer=l1_l2(l1=1e-4, l2=1e-4)),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])
model.summary()

# Otimizador e compilação do modelo
optimizer = Adam(learning_rate=1.61e-6)
model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=['acc'])

# Callbacks
#earlystop = EarlyStopping(monitor='val_acc', patience=10, restore_best_weights=True)
checkpoint = ModelCheckpoint(filepath='best_weights.h5', save_best_only=True, save_weights_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=3, min_lr=1.61e-8, verbose=1)
callbacks = [checkpoint, reduce_lr]

# Geradores de dados com aumento
datagen_train = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

datagen_validation = ImageDataGenerator(rescale=1./255)

train_generator = datagen_train.flow_from_directory(
    'D:/Users/Estevaos108/Desktop/Estevao Files/Imagens de Treino',
    target_size=(180, 180),
    batch_size=32,
    class_mode='binary'
)

validation_generator = datagen_validation.flow_from_directory(
    'D:/Users/Estevaos108/Desktop/Estevao Files/Imagens de Validacao',
    target_size=(180, 180),
    batch_size=32,
    class_mode='binary'
)

steps_per_epoch = int(np.ceil(train_generator.n / float(train_generator.batch_size)))
validation_steps = int(np.ceil(validation_generator.n / float(validation_generator.batch_size)))

# Treinamento do modelo
history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=100,
    validation_data=validation_generator,
    validation_steps=validation_steps,
    callbacks=callbacks,
    verbose=1,
    workers=4,
    use_multiprocessing=False
)

print("Média do Loss de Treino:", np.mean(history.history['loss']))

print("Média da Acurácia de Treino:", np.mean(history.history['acc']))

print("Média do Loss de Validação:", np.mean(history.history['val_loss']))

print("Média da Acurácia de Validação:", np.mean(history.history['val_acc']))

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Modelo de Acurácia')
plt.ylabel('Acurácia')
plt.xlabel('Epochs')
plt.legend(['Treino', 'Validação'], loc='lower right')
plt.show()

model.save('D:/Users/Estevaos108/Desktop/Estevao Files/modelo/best_weights.h5') # salvando modelo
