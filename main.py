import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))

from paddle_test_reader import PaddleOCR_read_pdf_and_predict_text
from query_generation_from_llm import SQLQueryGenerator

def main():

    ocr = PaddleOCR_read_pdf_and_predict_text(r"Z:\projects\SQL_query_from_PDF\data\test2.pdf")
    ocr.download_models()
    ocr.extract_models()
    ocr.fix_models()
    ocr.paddle_ocr_predict()
    
    print(f"OCR обработка завершена. Обработано страниц: {len(ocr.all_pages_results)}")
    print("Запуск генерации SQL запроса")
    
    chatbot = SQLQueryGenerator(schema_path="config.yml")  
    
    if ocr.all_pages_results and len(ocr.all_pages_results) > 0:
        sql_query = chatbot.generate_sql(ocr)
        
        print("Сгенерированный SQL запрос")
        print(sql_query)
        
if __name__ == "__main__":
    main()