FROM public.ecr.aws/lambda/python:3.9-x86_64

# Копируем необходимые файлы
COPY model.tflite .
COPY lambdatest.py .
COPY tflite_runtime-2.7.0-cp39-cp39-manylinux2014_x86_64.whl .

# Устанавливаем зависимости
RUN pip install keras_image_helper
RUN pip install tflite
RUN pip install tflite_runtime-2.7.0-cp39-cp39-manylinux2014_x86_64.whl
RUN pip install requests numpy Pillow

# Устанавливаем команду запуска вашего обработчика
CMD ["lambdatest.lambda_handler"]
