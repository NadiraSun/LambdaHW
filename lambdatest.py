import tflite_runtime.interpreter as tflite
import numpy as np
from PIL import Image
import requests
from io import BytesIO

# Загрузка модели и создание интерпретатора
interpreter = tflite.Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()

# Получение индекса ввода и вывода
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_index = input_details[0]['index']
output_index = output_details[0]['index']

# Определение лямбда-функции для предварительной обработки изображения
preprocess_input = lambda x: (x / 127.5) - 1.0

# Функция для выполнения инференса на изображении
def predict(url):
    # Получение изображения по URL
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))

    # Изменение размера изображения и преобразование в массив numpy
    img = img.resize((150, 150), Image.NEAREST)
    x = np.array(img, dtype='float32')
    X = np.array([x])

    # Предварительная обработка изображения
    X = preprocess_input(X)

    # Запуск инференса
    interpreter.set_tensor(input_index, X)
    interpreter.invoke()

    # Получение результатов
    output_data = interpreter.get_tensor(output_index)

    # Маппинг предсказаний на классы
    classes = ['dragon', 'dino']
    return dict(zip(classes, output_data[0]))

# Пример использования
url = 'https://formaxfun.com/wp-content/uploads/2023/12/novogodnie-kartinki-s-simvolom-2024-goda-zelyonym-derevyannym-drakonom_658776db4f744.jpeg' 
predictions = predict(url)
print(predictions)


def lambda_handler(event, context):
    url = event['url']
    preds = predict(url)
    return {
        'statusCode':200,
        'body': str(preds)
    }