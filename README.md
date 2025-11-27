# Программа по распознаванию текста из PDF и преобразование его в запрос к SQL базе данных (Paddle OCR + LLM)

References: 


https://habr.com/ru/articles/933634/
https://habr.com/ru/companies/bothub/articles/925632/


Программа принимает на вход PDF документ с содержанием русского/белорусского/украинского/английского языков, далее при помощи Paddle OCR распознает его содержимое. Содержимое документа добается на вход к одной из моделей на выбор: Phi2 (~ 3 млр. параметров) или Mistral7b (7 миллиардов параметров), после на выход выдается JSON-файл с запросами к базе данных согласно указанной пользователем схеме БД и промптом. Сочетает в себе элементы тонкой настройки и обращение к схеме БД пользователя.

В config.yaml содержит схему базы данных, необходимо указать свою

## Требования к установке

- OC Windows
- Python версии 3.12.8
- CUDA 12.8
- Poppler в PATH (добавить в переменную окружения Popper, для Windows архив скачать отсюда: https://github.com/oschwartz10612/poppler-windows/releases/) 

## Установка

1. Клонировать репозиторий:

```shell
git clone https://github.com/gromoveveryday/SQL_query_from_PDF.git
```

```shell
cd SQL_query_from_PDF
```

2. Создать и активировать виртуальное окружение:

```shell
python -m venv venv
```

```shell
venv\Scripts\activate
```

3. Установить/обновить зависимости:

```shell
pip install -r requirements.txt
```

## Запуск приложения

1. Запуск приложения:

```shell
python main.py
```

