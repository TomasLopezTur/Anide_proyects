from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Cargar dataset
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_data = datagen.flow_from_directory(
    "dataset_ovas",
    target_size=(128, 128),
    batch_size=32,
    class_mode="categorical",
    subset="training"
)

val_data = datagen.flow_from_directory(
    "dataset_ovas",
    target_size=(128, 128),
    batch_size=32,
    class_mode="categorical",
    subset="validation"
)

# Definir modelo CNN
modelo = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(128, 128, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation="relu"),
    Dense(3, activation="softmax")  # 3 categorías (Viva, Muerta, Cíclope)
])

modelo.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Entrenar modelo
modelo.fit(train_data, validation_data=val_data, epochs=10)

# Guardar modelo entrenado
modelo.save("clasificador_ovas.h5")
