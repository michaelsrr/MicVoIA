# Utiliza la imagen base de Python 3.10
FROM python:3.10

# Establece el directorio de trabajo en /app
WORKDIR /app

# Copia los archivos del directorio actual al directorio de trabajo del contenedor
COPY . /app

# Instala las dependencias (incluyendo PortAudio)
RUN pip install flask sounddevice soundfile tensorflow librosa audioread

# Expone el puerto 5000 para Flask
EXPOSE 5000

# Comando para ejecutar la aplicación Flask
CMD ["python", "app.py"]
