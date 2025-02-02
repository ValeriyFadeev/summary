# Используем Python 3.11
FROM python:3.11

# Устанавливаем рабочую директорию
WORKDIR /app
COPY ./requirements.txt /app/requirements.txt
# Обновляем pip
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r /app/requirements.txt
# Устанавливаем Uvicorn напрямую с GitHub
RUN pip install --no-cache-dir git+https://github.com/encode/uvicorn.git

# Копируем весь код проекта
COPY . .

# Открываем порт 8000
EXPOSE 8000

# Запускаем сервер
CMD ["uvicorn", "sum:app", "--host", "0.0.0.0", "--port", "8000"]
