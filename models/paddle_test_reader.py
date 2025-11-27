import os
import json
import tarfile
import requests
import yaml
import paddleocr
import cv2
import numpy as np
from pdf2image import convert_from_path

class PaddleOCR_read_pdf_and_predict_text:
    def __init__(self, pdf_path: str):

        self.pdf_path = pdf_path
        self.cache_tars = "../cache/tars"
        self.official_models = "../cache/official_models"
        base = os.path.splitext(os.path.basename(pdf_path))[0]
        self.output_dir = os.path.join(os.path.dirname(pdf_path), f"{base}_images")

        os.makedirs(self.cache_tars, exist_ok=True)
        os.makedirs(self.official_models, exist_ok=True)

        self.model_urls = {
            "rec": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/eslav_PP-OCRv5_mobile_rec_infer.tar",
            "det": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_mobile_det_infer.tar"
        }

        self.tar_paths = {}
        self.model_dirs = {}

    def download_models(self):
        for mtype, url in self.model_urls.items():
            fname = os.path.basename(url)
            tar_path = os.path.join(self.cache_tars, fname)

            if not os.path.exists(tar_path):
                print(f"Скачивание {fname} ...")
                self._download_file(url, tar_path)
            else:
                print(f"Уже скачан: {fname}")

            self.tar_paths[mtype] = tar_path

    def _download_file(self, url, save_path):
        try:
            r = requests.get(url, stream=True)
            r.raise_for_status()

            with open(save_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True
        except:
            return False

    def extract_models(self):
        for mtype, tar_path in self.tar_paths.items():
            folder = os.path.basename(tar_path).replace(".tar", "")
            extract_dir = os.path.join(self.official_models, folder)

            if not os.path.exists(extract_dir):
                print(f"Распаковка {folder} ...")
                self._extract_tar(tar_path, extract_dir)
            else:
                print(f"Уже распакована: {folder}")

            self.model_dirs[mtype] = extract_dir

    def _extract_tar(self, src, dst):
     
        os.makedirs(dst, exist_ok=True)

        with tarfile.open(src, "r") as tar:

            members = tar.getmembers()
            top_level_dirs = set(m.name.split("/")[0] for m in members)

            if len(top_level_dirs) == 1:
                top = list(top_level_dirs)[0]

                for m in members:
                    m_path = m.name.split("/", 1)[1] if "/" in m.name else None
                    if not m_path:
                        continue  # пропускаем корневую папку

                    m.name = m_path  # переопределяем путь
                    tar.extract(m, dst)
            else:

                tar.extractall(path=dst)

    def fix_models(self):
       
        self._fix_inference_yml(
            self.model_dirs.get("rec"),
            old_name="eslav_PP-OCRv5_mobile_rec", # PaddleOCR настолько классно сделана, что другие модели кроме основной на 4 языка, она просто по названию не пропускает и выдает ошибку, единственный выход - прям в конфиге переименовать
            new_name="PP-OCRv5_server_rec"
        )

        self._fix_inference_yml(
            self.model_dirs.get("det"),
            old_name="PP-OCRv5_mobile_det", 
            new_name="PP-OCRv5_server_det"
        )

    def _fix_inference_yml(self, model_dir, old_name, new_name):
        if not model_dir:
            return

        yml_path = os.path.join(model_dir, "inference.yml")
        if not os.path.exists(yml_path):
            return

        with open(yml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        current = data.get("Global", {}).get("model_name", None)

        if current == new_name:
            print(f"Уже исправлено: {new_name}")
            return

        if current == old_name:
            print(f"Исправление model_name: {old_name} → {new_name}")
            data["Global"]["model_name"] = new_name
        else:
            print(f"[i] Нестандартное имя model_name, не изменилось: {current}")
            return

        with open(yml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
    
    def paddle_ocr_predict(self):
        self.images = convert_from_path(self.pdf_path)
        
        ocr_model = paddleocr.PaddleOCR(
        use_textline_orientation=False,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        text_detection_model_dir=self.model_dirs['det'],
        text_recognition_model_dir=self.model_dirs['rec']
        )

        self.all_pages_results = []

        for page_idx, pil_img in enumerate(self.images, 1):
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            result = ocr_model.predict(img)
            self.all_pages_results.append(result)