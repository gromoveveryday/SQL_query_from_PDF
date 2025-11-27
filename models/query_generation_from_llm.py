import yaml
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class SQLQueryGenerator:
    def __init__(self, schema_path, model_name="microsoft/phi-2"):  # или mistralai/Mistral-7B-v0.1
        
        self.model_name = model_name

        # Загрузка схемы базы
        with open(schema_path, "r", encoding="utf-8") as f:
            self.schema = yaml.safe_load(f)

        # Настройка trust_remote_code
        # Phi-2 — требует
        # Mistral — НЕ требует
        trust_code = "phi" in model_name.lower()

        # Загрузка токенайзера
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_code
        )

        # У Mistral нет pad_token → устанавливаем вручную
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Загрузка модели
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=r"..\\cache",
            torch_dtype=torch.float16,
            device_map="cuda",
            trust_remote_code=trust_code
        )

        # Настройки генерации SQL
        self.gen_config = {
            "temperature": 0.1,
            "top_p": 0.7,
            "max_new_tokens": 200,
            "repetition_penalty": 1.15,
            "no_repeat_ngram_size": 2
        }

        # Few-shot примеры
        self.few_shots = """
Пример 1.
Текст: ["3A", "11,13", "Южный федеральный округ", "01.11.2025"]
Схема таблицы call_data: id: INTEGER
    category_name: INTEGER
    district_name: TEXT
    call_date: DATE
    price: FLOAT

SQL: INSERT INTO call_data (category_name, district_name, price, call_date)
    VALUES ("3А", "Южный федеральный округ", "11.13", "01.11.2025";

Пример 2.
Текст: ["12A", "10,67", "Центральный федеральный округ", "01.11.2025", "Алюминий электротех", "177,5"]
Схема таблицы call_data: id: INTEGER
    category_name: INTEGER
    district_name: TEXT
    call_date: DATE
    price: FLOAT

SQL: INSERT INTO call_data (category_name, district_name, price, call_date)
    VALUES ("12А", "Центральный федеральный округ", "10.67", "01.11.2025"),
    VALUES ("Алюминий электротех", "Центральный федеральный округ", "177.5", "01.11.2025");
"""

    # Генерация SQL на основе текста OCR
    def generate_sql(self, ocr_object):

        # Собираем текст всех страниц (если pdf на нескольк страниц)
        pages_text = []
        for page in ocr_object.all_pages_results:
            if page and isinstance(page, list) and "rec_texts" in page[0]:
                pages_text.extend(page[0]["rec_texts"])
        
        ocr_text = "\n".join(pages_text)

        # Схема БД в текст
        schema_text = yaml.dump(self.schema, allow_unicode=True)

        # Формируем промпт
        prompt = f"""
Ты — система, которая получает текст со страницы PDF и должна
на основе него сформировать корректный SQL-запрос согласно схеме.

Схема базы данных:
{schema_text}

Найди в тексте категорию лома, цену, дату, регион.
Свяжи их с таблицей call_data.

Если региона нет — district_id = NULL.

Генерируй только SQL, без комментариев.

--- Few-shot Examples ---
{self.few_shots}

--- TEXT INPUT ---
{ocr_text}

SQL:
"""
        # Генерация
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.gen_config["max_new_tokens"],
                temperature=self.gen_config["temperature"],
                top_p=self.gen_config["top_p"],
                repetition_penalty=self.gen_config["repetition_penalty"],
                no_repeat_ngram_size=self.gen_config["no_repeat_ngram_size"],
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True
            )

        text = self.tokenizer.decode(output[0], skip_special_tokens=True)

        # Достаём только SQL часть
        if "SQL:" in text:
            text = text.split("SQL:", 1)[-1].strip()

        lines = text.split("\n")
        cleaned = []
        for line in lines:
            if line.strip().upper().startswith("VALUES") and not cleaned:
                continue
            cleaned.append(line)

        text = "\n".join(cleaned).strip()

        return text
