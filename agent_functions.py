import pandas as pd
import os
import pdfplumber
import pymupdf4llm
import fitz
import json
import numpy as np
import asyncio
import uuid
import re
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from sentence_transformers import SentenceTransformer
from langchain_community.tools.tavily_search import TavilySearchResults
from utils import extract_price_from_text, assess_data_quality, parse_tavily_results
from playwright.async_api import async_playwright, TimeoutError
from serpapi import GoogleSearch
from openai import OpenAI
from docx import Document
import time

import google.generativeai as genai
from dotenv import load_dotenv, dotenv_values

class ModelLoader:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        print("🔧 Инициализация моделей...")
        load_dotenv()
        
        self.qdrant_client = QdrantClient(path="./qdrant_data")
        self.collection_name = "risky_tenders"
        self.collection_name_v2 = "approved_risks"
        
        self.embedding_model = SentenceTransformer('intfloat/multilingual-e5-large')
        
        gemini_key = os.getenv("GOOGLE_API_KEY")
        if not gemini_key:
            raise ValueError("Установите GOOGLE_API_KEY в .env файле!")
        genai.configure(api_key=gemini_key)
        self.gemini_model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                temperature=0.1
            )
        )
        
        openai_key = os.getenv("OPENAI_API_KEY")
        self.openai_client = OpenAI(api_key=openai_key) if openai_key else None
        
        self.serp_api_key = os.getenv("SERP_API_KEY")
        self.tavily_key = os.getenv("TAVILY_API_KEY")
        
        self._initialized = True
        print("✅ Все модели загружены!\n")

models = ModelLoader()

# ============================================================================
# ИСПРАВЛЕННАЯ ФУНКЦИЯ: search_download_and_parse
# Теперь она ОБЫЧНАЯ async функция, которую будет вызывать wrapper
# ============================================================================
# ============================================================================
# ФУНКЦИЯ 3: PLAYWRIGHT SCRAPER (ASYNC)
# ============================================================================

async def search_download_and_parse_async(SEARCH_TERM: str, target_count: int = 3):
    """
    Внутренняя async функция для Playwright.
    НЕ ВЫЗЫВАЙТЕ НАПРЯМУЮ! Используйте search_download_and_parse()
    """
    BASE_SEARCH_URL = "https://eep.mitwork.kz/ru/publics/lots?filter[submit]=&filter[search]="
    DOMAIN = "https://eep.mitwork.kz"
    LOT_LINK_SELECTOR = 'a.word-break'
    DOWNLOAD_DIR = os.path.join(os.getcwd(), "downloads")
    MAX_ATTEMPTS_LIMIT = target_count * 3

    if not os.path.exists(DOWNLOAD_DIR):
        os.makedirs(DOWNLOAD_DIR)

    print(f"🔍 Поиск: '{SEARCH_TERM}'")
    print(f"🎯 Цель: {target_count} документов")

    async def parse_tables_context(page):
        data = {}
        table_ids = ["#w0", "#w3"]
        
        for t_id in table_ids:
            try:
                if await page.locator(t_id).count() > 0:
                    rows = page.locator(f"{t_id} tr")
                    count = await rows.count()
                    
                    for i in range(count):
                        row = rows.nth(i)
                        th = row.locator("th")
                        td = row.locator("td")
                        
                        if await th.count() > 0 and await td.count() > 0:
                            key = await th.inner_text()
                            val = await td.inner_text()
                            data[key.strip()] = val.strip()
            except Exception as e:
                print(f"   ⚠️ Ошибка таблицы {t_id}: {e}")
        
        return data

    async def process_lot_page(page, lot_url, index):
        print(f"\n📄 [{index}] Проверка лота: {lot_url}")
        try:
            await page.goto(lot_url, wait_until="domcontentloaded", timeout=30000)
            
            try:
                await page.wait_for_selector('table', timeout=5000)
            except:
                print(f"   ⚠️ Таблица не найдена.")
                return None

            patterns_to_try = [
                r"технически.*спецификац", r"тех[\.\s]+спец",
                r"приложение\s*[№]?\s*2", r"приложение\s*[№]?\s*1",
                r"спецификац"
            ]

            download_button = None
            all_rows = page.locator("tr")
            row_count = await all_rows.count()

            for pattern in patterns_to_try:
                for i in range(row_count):
                    row = all_rows.nth(i)
                    row_text = await row.inner_text()
                    if re.search(pattern, row_text, re.IGNORECASE):
                        btn = row.locator("a:has(span.glyphicon-download-alt)")
                        if await btn.count() > 0:
                            download_button = btn.first
                            break
                if download_button:
                    break

            if not download_button:
                print(f"   ❌ Файл не найден.")
                return None

            print(f"   ✅ Скачиваем файл...")
            
            async with page.expect_download(timeout=30000) as download_info:
                await download_button.click(force=True)

            download = await download_info.value
            safe_filename = f"lot_{index}_{download.suggested_filename}"
            save_path = os.path.join(DOWNLOAD_DIR, safe_filename)
            await download.save_as(save_path)
            
            context_data = await parse_tables_context(page)
            
            return {
                "file_path": save_path,
                "source_url": lot_url,
                "context": context_data
            }

        except Exception as e:
            print(f"   ❌ Ошибка: {e}")
            return None

    results = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(accept_downloads=True)
        page = await context.new_page()

        full_search_url = BASE_SEARCH_URL + SEARCH_TERM
        await page.goto(full_search_url, wait_until="domcontentloaded", timeout=30000)

        try:
            await page.wait_for_selector(LOT_LINK_SELECTOR, timeout=8000)
        except:
            print("❌ Лоты не найдены.")
            await browser.close()
            return []

        lot_elements = page.locator(LOT_LINK_SELECTOR)
        count_found = await lot_elements.count()
        
        links_to_check = []
        for i in range(min(count_found, MAX_ATTEMPTS_LIMIT)):
            href = await lot_elements.nth(i).get_attribute("href")
            if href:
                url = href if href.startswith("http") else DOMAIN + href
                links_to_check.append(url)

        print(f"📋 Найдено {count_found} лотов. Проверяем {len(links_to_check)}...")

        for i, link in enumerate(links_to_check):
            if len(results) >= target_count:
                break
            
            data = await process_lot_page(page, link, i + 1)
            if data:
                results.append(data)

        await browser.close()

    print(f"🎁 Собрано: {len(results)} документов")
    return results


# ============================================================================
# ПУБЛИЧНАЯ ФУНКЦИЯ: ИСПОЛЬЗУЙТЕ ЭТУ!
# ============================================================================
async def search_download_and_parse(SEARCH_TERM: str, target_count: int = 3):
    """
    ✅ ПРАВИЛЬНАЯ async функция для use в main.py
    Работает напрямую в async контексте LangGraph!
    
    Args:
        SEARCH_TERM: Поисковый запрос
        target_count: Количество документов
        
    Returns:
        List[dict]: Список скачанных документов
    """
    return await search_download_and_parse_async(SEARCH_TERM, target_count)

# Глобальный экземпляр загрузчика
models = ModelLoader()
def LoadDFinQdrant(df: pd.DataFrame):
    """
    Загружает рискованные контракты из DataFrame в Qdrant.
    Создает векторные эмбеддинги для семантического поиска.
    """
    print("Подготовка данных для Qdrant...")
    
    # ФИЛЬТР: Берем только контракты с рисками
    risky_df = df[df['количество_рисков'] >= 1].copy()
    risky_df = risky_df.dropna(subset=['описание_чистое'])
    
    print(f"Найдено рискованных контрактов: {len(risky_df)}")
    
    # Создаем/пересоздаем коллекцию
    models.qdrant_client.recreate_collection(
        collection_name=models.collection_name,
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
    )
    # Заливка батчами для скорости
    batch_size = 100
    points = []
    
    print("Векторизация и загрузка...")
    
    for idx, row in risky_df.iterrows():
        text = row['описание_чистое']
        
        # Векторизуем (префикс 'passage:' для e5 моделей)
        vector = models.embedding_model.encode(f"passage: {text}").tolist()
        
        # Формируем метаданные
        payload = {
            "contract_id": str(row['номер_договора']),
            "amount": float(row['сумма']),
            "buyer": str(row['заказчик']),
            "risk_count": int(row['количество_рисков']),
            "risk_overprice": int(row.get('флаг_завышения_цены', 0)),
            "risk_split": int(row.get('флаг_дробление_закупок', 0)),
            "original_text": text
        }
        
        points.append(PointStruct(
            id=str(uuid.uuid4()), 
            vector=vector, 
            payload=payload
        ))
        
        # Когда набрали батч — отправляем
        if len(points) >= batch_size:
            models.qdrant_client.upsert(
                collection_name=models.collection_name, 
                points=points
            )
            points = []
            print(f"   Обработано: {idx}...")
    
    # Доливаем остатки
    if points:
        models.qdrant_client.upsert(
            collection_name=models.collection_name, 
            points=points
        )
    
    print("База рисков успешно создана!")


# ============================================================================
# ФУНКЦИЯ 2: ПРОВЕРКА ПОХОЖИХ РИСКОВ В ИСТОРИИ
# ============================================================================
def check_similarity_risk(new_tender_description: str):
    """
    Ищет похожие схемы в базе прошлых нарушений.
    Возвращает список похожих контрактов с указанием рисков.
    """
    print(f"🔍 Ищу похожие риски для: {new_tender_description[:100]}...")
    
    # Векторизуем запрос (префикс 'query:' для поиска)
    query_vector = models.embedding_model.encode(
        f"query: {new_tender_description}"
    ).tolist()
    
    # Поиск в Qdrant
    search_result = models.qdrant_client.query_points(
        collection_name=models.collection_name,
        query=query_vector,
        limit=5,
        score_threshold=0.82  # Порог сходства 82%
    )
    
    if not search_result:
        print("  Похожих рисков в истории не найдено.")
        return None
    # Поиск в бэкап базе
    search_result_backup = models.qdrant_client.query_points(
        collection_name=models.collection_name_v2,
        query=query_vector,
        limit=3, # Для быстроты
        score_threshold=0.82        
    )
    # Формируем результат
    found_risks = []
    for hit in search_result.points:
        risk_info = {
            "similarity": round(hit.score * 100, 1),
            "past_contract": hit.payload['contract_id'],
            "past_buyer": hit.payload['buyer'],
            "past_amount": hit.payload['amount'],
            "risk_reason": []
        }
        
        # Расшифровка типов рисков
        if hit.payload['risk_overprice']:
            risk_info['risk_reason'].append("Завышение цены")
        if hit.payload['risk_split']:
            risk_info['risk_reason'].append("Дробление закупки")
        
        found_risks.append(risk_info)
    
    print(f"   ⚠️ Найдено {len(found_risks)} похожих случаев!")
    return found_risks


# ============================================================================
# ФУНКЦИЯ 3: УЛУЧШЕННЫЙ СКРЕЙПИНГ ДОКУМЕНТОВ С PLAYWRIGHT
# ============================================================================

async def search_download_and_parse(SEARCH_TERM: str, target_count: int = 3):
    """
    Асинхронная функция поиска и скачивания документов.
    ✅ ПОЛНОСТЬЮ ПЕРЕПИСАНА ДЛЯ ASYNC
    """
    BASE_SEARCH_URL = "https://eep.mitwork.kz/ru/publics/lots?filter[submit]=&filter[search]="
    DOMAIN = "https://eep.mitwork.kz"
    LOT_LINK_SELECTOR = 'a.word-break'
    DOWNLOAD_DIR = os.path.join(os.getcwd(), "downloads")
    
    MAX_ATTEMPTS_LIMIT = target_count * 3 

    if not os.path.exists(DOWNLOAD_DIR):
        os.makedirs(DOWNLOAD_DIR)

    print(f"🔍 Поиск: '{SEARCH_TERM}'")
    print(f"🎯 Цель: {target_count} документов")

    async def parse_tables_context(page):
        """Парсит таблицы асинхронно."""
        data = {}
        table_ids = ["#w0", "#w3"]
        
        for t_id in table_ids:
            try:
                if await page.locator(t_id).count() > 0:
                    rows = page.locator(f"{t_id} tr")
                    count = await rows.count()
                    
                    for i in range(count):
                        row = rows.nth(i)
                        th = row.locator("th")
                        td = row.locator("td")
                        
                        if await th.count() > 0 and await td.count() > 0:
                            key = await th.inner_text()
                            val = await td.inner_text()
                            data[key.strip()] = val.strip()
            except Exception as e:
                print(f"   ⚠️ Ошибка таблицы {t_id}: {e}")
        
        return data

    async def process_lot_page(page, lot_url, index):
        """Обработка одного лота асинхронно."""
        print(f"\n📄 [{index}] Проверка лота: {lot_url}")
        try:
            await page.goto(lot_url, wait_until="domcontentloaded", timeout=30000)
            
            try:
                await page.wait_for_selector('table', timeout=5000)
            except:
                print(f"   ⚠️ Таблица не найдена.")
                return None

            patterns_to_try = [
                r"технически.*спецификац", r"тех[\.\s]+спец", 
                r"приложение\s*[№]?\s*2", r"приложение\s*[№]?\s*1", 
                r"спецификац"
            ]

            download_button = None
            all_rows = page.locator("tr")
            row_count = await all_rows.count()

            for pattern in patterns_to_try:
                for i in range(row_count):
                    row = all_rows.nth(i)
                    row_text = await row.inner_text()
                    if re.search(pattern, row_text, re.IGNORECASE):
                        btn = row.locator("a:has(span.glyphicon-download-alt)")
                        if await btn.count() > 0:
                            download_button = btn.first
                            break
                if download_button: 
                    break

            if not download_button:
                print(f"   ❌ Файл не найден.")
                return None

            print(f"   ✅ Скачиваем файл...")
            
            async with page.expect_download(timeout=30000) as download_info:
                await download_button.click(force=True)

            download = await download_info.value
            safe_filename = f"lot_{index}_{download.suggested_filename}"
            save_path = os.path.join(DOWNLOAD_DIR, safe_filename)
            await download.save_as(save_path)
            
            context_data = await parse_tables_context(page)
            
            return {
                "file_path": save_path,
                "source_url": lot_url,
                "context": context_data
            }

        except Exception as e:
            print(f"   ❌ Ошибка: {e}")
            return None

    results = []

    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(accept_downloads=True)
            page = await context.new_page()

            full_search_url = BASE_SEARCH_URL + SEARCH_TERM
            await page.goto(full_search_url, wait_until="domcontentloaded", timeout=30000)

            try:
                await page.wait_for_selector(LOT_LINK_SELECTOR, timeout=8000)
            except:
                print("❌ Лоты не найдены.")
                await browser.close()
                return []

            lot_elements = page.locator(LOT_LINK_SELECTOR)
            count_found = await lot_elements.count()
            
            links_to_check = []
            for i in range(min(count_found, MAX_ATTEMPTS_LIMIT)):
                href = await lot_elements.nth(i).get_attribute("href")
                if href:
                    url = href if href.startswith("http") else DOMAIN + href
                    links_to_check.append(url)

            print(f"📋 Найдено {count_found} лотов. Проверяем {len(links_to_check)}...")

            for i, link in enumerate(links_to_check):
                if len(results) >= target_count:
                    break
                
                data = await process_lot_page(page, link, i + 1)
                if data:
                    results.append(data)

            await browser.close()
            
    except Exception as e:
        print(f"❌ Ошибка Playwright: {e}")
        return []

    print(f"🎁 Собрано: {len(results)} документов")
    return results

# ============================================================================
# ФУНКЦИЯ 4: ИЗВЛЕЧЕНИЕ ТЕКСТА ИЗ PDF
# ============================================================================
def extract_data_from_pdf(pdf_path: str):
    """
    Извлекает текст и таблицы из PDF с сохранением структуры.
    Использует pdfplumber для точного извлечения таблиц.
    """
    full_content = ""
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text_content = f"\n--- Страница {i+1} ---\n"
                
                # Извлекаем таблицы
                tables = page.extract_tables()
                
                if tables:
                    text_content += "ТАБЛИЦЫ:\n"
                    for table in tables:
                        for row in table:
                            cleaned_row = [
                                str(cell).replace('\n', ' ') if cell else "" 
                                for cell in row
                            ]
                            text_content += "| " + " | ".join(cleaned_row) + " |\n"
                        text_content += "\n"
                
                # Извлекаем текст
                raw_text = page.extract_text()
                if raw_text:
                    text_content += "\nТЕКСТ:\n" + raw_text
                
                full_content += text_content
        
        return full_content
    
    except Exception as e:
        print(f"❌ Ошибка чтения PDF: {e}")
        return None


# ============================================================================
# ФУНКЦИЯ 5: АНАЛИЗ ДОКУМЕНТА ЧЕРЕЗ GEMINI (УЛУЧШЕННЫЙ МЕТОД)
# ============================================================================
def extract_facts_for_researcher(tender_object: dict) -> dict:
    """
    Превращает PDF/DOCX + Контекст таблицы в сухой JSON с фактами.
    Никаких вердиктов, только данные для будущего анализа.
    """
    file_path = tender_object.get("file_path")
    table_context = tender_object.get("context", {})
    
    print(f"Добываю факты из: {os.path.basename(file_path)}")

    # 1. Склеиваем контекст таблицы (это наш "Ground Truth" по цифрам)
    context_str = "\n".join([f"{k}: {v}" for k, v in table_context.items()])

    # 2. Читаем файл (поддержка docx/pdf)
    file_text = ""
    if os.path.exists(file_path):
        try:
            ext = os.path.splitext(file_path)[1].lower()
            if ext == ".pdf":
                doc = fitz.open(file_path)
                for page in doc: file_text += page.get_text() + "\n"
                doc.close()
            elif ext in [".docx", ".doc"]:
                doc = Document(file_path)
                file_text = "\n".join([p.text for p in doc.paragraphs])
        except Exception as e:
            print(f"⚠️ Ошибка чтения файла: {e}")
            file_text = "Текст файла недоступен."
    
    # 3. ✅ ИСПРАВЛЕННАЯ Схема данных "FACT SHEET"
    FACT_SHEET_SCHEMA = {
        "type": "object",
        "properties": {
            "lot_id": {"type": "string", "description": "Номер лота или название файла"},
            "meta_info": {
                "type": "object",
                "properties": {
                    "zakazchik": {"type": "string"},
                    "organizator": {"type": "string"},
                    "kontaktnye_dannye": {"type": "string"},
                    "predmet_zakupki": {"type": "string", "description": "Краткое название предмета закупки"}
                }
            },
            "finance": {
                "type": "object",
                "properties": {
                    "budget_total": {"type": "number", "description": "Сумма цифрами"},
                    "currency": {"type": "string"},
                    "usloviya_oplaty": {"type": "string", "description": "Например: 30% предоплата, 70% по факту"},
                    "obespehenie_zayavki": {"type": "string"}
                }
            },
            "logistics": {
                "type": "object",
                "properties": {
                    "mesto_postavki": {"type": "string", "description": "Точный адрес или город"},
                    "srok_postavki": {"type": "string", "description": "Дней или конкретная дата"},
                    "incoterms": {"type": "string", "description": "DDP, EXW и т.д."}
                }
            },
            "tech_specs": {
                "type": "array",
                "description": "Список конкретных технических требований к товару/услуге",
                "items": {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string", "description": "СТРОГО ОПРЕДЕЛИ ЧТО ЭТО? (GOODS OR SERVICES OR CONSTRUCTION OR UNKNOWN OR OTHER?)"},
                        "parameter": {"type": "string", "description": "Название параметра (напр. Процессор)"},
                        "value": {"type": "string", "description": "Требуемое значение (напр. Core i7)"},
                        "is_critical": {"type": "boolean", "description": "Является ли это строгим требованием"}
                    }
                }
            },
            "legal_requirements": {
                "type": "array",
                "description": "Требуемые лицензии, сертификаты (ISO, СТ-KZ)",
                "items": {"type": "string"}
            }
        }
    }

    # 4. Промпт "Дата-Инженер"
    PROMPT = f"""
    Ты — Data Extractor. Твоя задача — извлечь структурированные данные для базы данных.
    НЕ делай выводов. НЕ пиши эссе. Только извлечение сущностей.

    Используй два источника:
    1. МЕТА-ДАННЫЕ (Приоритет по ценам и срокам):
    {context_str}

    2. ТЕКСТ ДОКУМЕНТА (Технические детали):
    {file_text[:60000]}

    Если информации нет, пиши null или "Не указано".
    В поле 'tech_specs' выдели главные технические характеристики товара.
    
    ВАЖНО: В поле 'type' внутри tech_specs определи тип закупки:
    - GOODS (товары, оборудование, техника)
    - SERVICES (услуги, работы, обслуживание)
    - CONSTRUCTION (строительство, ремонт)
    - UNKNOWN (если не можешь определить)
    """

    try:
        response = models.gemini_model.generate_content(
            PROMPT,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=FACT_SHEET_SCHEMA
            )
        )
        
        result = json.loads(response.text)
        
        # ✅ Добавляем ID лота из имени файла
        if not result.get("lot_id"):
            result["lot_id"] = os.path.basename(file_path)
        
        print(f"   ✅ Факты извлечены успешно")
        return result
    
    except json.JSONDecodeError as e:
        print(f"   ❌ Ошибка парсинга JSON: {e}")
        return {"error": f"JSON parse error: {e}", "file": file_path}
    
    except Exception as e:
        print(f"   ❌ Ошибка Gemini: {e}")
        return {"error": str(e), "file": file_path}
# ============================================================================
# ФУНКЦИЯ 6: РЕЗЕРВНЫЙ АНАЛИЗ ЧЕРЕЗ GPT-4o-mini
# ============================================================================
def analyze_procurement_doc_gpt(text: str):
    """
    Запасной вариант анализа через OpenAI GPT-4o-mini.
    Используется если Gemini недоступен или дал плохой результат.
    """
    if not models.openai_client:
        return json.dumps({"error": "OpenAI клиент не инициализирован"})
    
    system_prompt = """
    *Ты аналитик закупок. Проанализируй документ (тендер/спецификация/договор).*
    Найди данные в тексте и таблицах.
    Текст может содержать Markdown-таблицы.
    Если что-то не найдено — пиши "Не указано".
    Если цена в тенге — оставляй KZT, если в рублях — RUB.
    **Верни JSON строго такой структуры:**
    {
        "type": "object",
        "properties": {
            "analiz_dokumenta": {
                "type": "object",
                "properties": {
                    "predmet_zakupki": {"type": "string"},
                    "tsena_obschaya": {"type": "string"},
                    "tsena_za_edinitsu": {"type": "string"},
                    "sroki_postavki_nachalo": {"type": "string"},
                    "sroki_postavki_konec": {"type": "string"},
                    "klyuchevye_trebovaniya": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "osobye_usloviya": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                },
                "required": ["predmet_zakupki", "tsena_obschaya"]
            }
        },
        "required": ["analiz_dokumenta"]
    }
    """
    
    truncated_text = text[:100000]
    
    try:
        response = models.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Документ:\n{truncated_text}"}
            ],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        return json.dumps({"error": f"Ошибка OpenAI: {e}"})


# ============================================================================
# ФУНКЦИЯ 7: ПОИСК РЫНОЧНЫХ ЦЕН ЧЕРЕЗ SERPAPI
# ============================================================================
def search_market_prices(query: str):
    """
    Ищет товар в Google Shopping и возвращает найденные цены.
    """
    print(f"🔎 Поиск рыночных цен: {query}")
    
    if not models.serp_api_key:
        print("   ⚠️ SerpAPI ключ отсутствует")
        return []
    
    params = {
        "q": query,
        "location": "Kazakhstan",
        "hl": "ru",
        "gl": "kz",
        "num": 10,
        "api_key": models.serp_api_key
    }
    
    try:
        search = GoogleSearch(params)
        results = search.get_dict()
        market_data = []
        
        # Проверяем Google Shopping
        if "shopping_results" in results:
            for item in results["shopping_results"]:
                price = item.get("price")
                if price:
                    market_data.append({
                        "source": item.get("source"),
                        "title": item.get("title"),
                        "price": price
                    })
        
        # Проверяем органическую выдачу
        if "organic_results" in results:
            for item in results["organic_results"]:
                if "rich_snippet" in item and "top" in item["rich_snippet"]:
                    detected_price = item["rich_snippet"]["top"].get(
                        "detected_extensions", {}
                    ).get("price")
                    if detected_price:
                        market_data.append({
                            "source": item.get("link"),
                            "title": item.get("title"),
                            "price": f"{detected_price} (из описания)"
                        })
        
        print(f"   ✅ Найдено {len(market_data)} цен")
        return market_data
    
    except Exception as e:
        print(f"   ❌ Ошибка SerpAPI: {e}")
        return []


# ============================================================================
# ФУНКЦИЯ 8?: АУДИТ ЦЕНЫ ЧЕРЕЗ AI #TODO:ЭТИ ФУНКЦИИ ЕСЛИ НЕ ОШИБАЮСЬ УЖЕ ВСТРОЕНЫ В RESEARCHGPT ЧТО ОБЛЕГЧИТ НАМ ЖИЗНЬ
# ============================================================================
def audit_price_with_ai(tender_json: str, market_data: list):
    """
    Сравнивает цену тендера с рыночными данными через Gemini.
    """
    print("💰 Аудит цен через AI...")
    
    tender_data = json.loads(tender_json)
    item_name = tender_data["analiz_dokumenta"]["predmet_zakupki"]
    tender_price = tender_data["analiz_dokumenta"]["tsena_za_edinitsu"]
    specs = tender_data["analiz_dokumenta"].get("osobye_usloviya", [])
    
    # Формируем описание рынка
    if not market_data:
        market_text = "Прямых цен в Google не найдено. Используй внутренние знания."
    else:
        market_text = ""
        for m in market_data:
            market_text += f"- {m['title']}: {m['price']} ({m['source']})\n"
    
    prompt = f"""
    ТЫ - АУДИТОР ГОСЗАКУПОК.
    
    ДАННЫЕ ТЕНДЕРА:
    Товар: {item_name}
    Цена за единицу: {tender_price}
    Спецификация: {specs}
    
    ДАННЫЕ РЫНКА:
    {market_text}
    
    ЗАДАЧА:
    1. Приведи цены к KZT (1 USD ≈ 450 KZT, 1 RUB ≈ 5 KZT).
    2. Сравни цену тендера с рынком. Учитывай комплектацию и условия.
    3. Дай вердикт.
    
    ФОРМАТ ОТВЕТА (JSON):
    {
        "market_price_range": "Диапазон рыночных цен в KZT",
        "tender_price_kzt": "Цена тендера (число)",
        "deviation_percent": "Отклонение в процентах (пример: +20%)",
        "verdict": "НОРМА / ЗАВЫШЕНО / ЗАНИЖЕНО / ТРЕБУЕТ ПРОВЕРКИ",
        "reasoning": "Краткое обоснование"
    }
    """
    
    try:
        response = models.gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return json.dumps({"error": f"Ошибка аудита: {e}"})


# ============================================================================
# ФУНКЦИЯ 9: ФИНАЛЬНЫЙ АНАЛИЗ #TODO: Заменить на функцию GPT-Researcher КОТОРАЯ БУДЕТ ДЕЛАТЬ ВЫВОД ДУМАЮ ЛУЧШЕ ВСЕГО ЭТО СДЕЛАТЬ В main_langgraph.py вместо FinalAnalysis
# ============================================================================
def FinalAnalysis(tender_json: str, market_audit: str, history_matches: list):
    """
    Финальный этап: синтез всех данных в итоговый вердикт.
    """
    print("🎯 Финальный анализ...")
    
    final_prompt = f"""
    ТЫ — ЭЛИТНЫЙ АУДИТОР ГОСЗАКУПОК.
    Выяви коррупцию, завышение цен и риски.
    
    ДАННЫЕ:
    1. ТЕНДЕР: {tender_json}
    2. АУДИТ РЫНКА: {market_audit}
    3. ИСТОРИЯ РИСКОВ: {json.dumps(history_matches, ensure_ascii=False)}
    
    ИНСТРУКЦИЯ:
    - ШАГ 1: Сравни цену тендера с рынком (>10% отклонение = риск)
    - ШАГ 2: Проверь историю (similarity >80% = копия схемы)
    - ШАГ 3: Сформируй вердикт
    
    ФОРМАТ ОТВЕТА (JSON):
    {{
        "tender_summary": {{
            "product": "...",
            "declared_price": 100000,
            "market_price_avg": 50000
        }},
        "risk_factors": [
            {{
                "type": "PRICE_OVERESTIMATION",
                "severity": "HIGH",
                "description": "Цена завышена на 100%"
            }}
        ],
        "conclusion_text": "Подробный анализ в формате Markdown..."
    }}
    """
    
    try:
        response = models.gemini_model.generate_content(final_prompt)
        return response.text
    except Exception as e:
        return json.dumps({"error": f"Ошибка финального анализа: {e}"})
# ============================================================================
# ФУНКЦИЯ 10: ДОБАВЛЕНИЕ КОРРЕКТНОГО ВЫВОДА В РЕЗЕРВНУЮ БАЗУ (ЕСЛИ ПОЛЬЗОВАТЕЛЬ ПОДТВЕРЖДАЕТ ВЕРДИКТ) 
# ============================================================================
def add_query_inrebase(final_verdict):
    tender_data = json.loads(final_verdict)
    text_for_vector = (
        f"passage: Товар: {tender_data['tender_summary']['product']}. "
        f"Вердикт: {tender_data['risk_factors']['description']}. "
        f"Причина: {tender_data['conclusion_text']}"
    )
    
    # Генерируем вектор
    vector = models.encode(text_for_vector).tolist()
    point_id = str(uuid.uuid4())
    
    # Записываем
    models.qdrant_client.upsert(
        collection_name=models.collection_name_v2,
        points=[
            models.PointStruct(
                id=point_id,
                vector=vector,
                payload=tender_data  # Кладем весь JSON в payload, чтобы видеть его при поиске
            )
        ]
    )
    print(f"✅ Вердикт загружен успешно! ID: {point_id}")
# ============================================================================
# ФУНКЦИЯ 11: ДОБАВЛЕНИЕ ОБРАБОТЧИКА DOCX ЕСЛИ ДОКУМЕНТ ОКАЖЕТСЯ НЕ PDF 
# ============================================================================
def docx_processing(docx_path):
    doc =  Document(docx_path)
    full_text = []
    for para in doc.paragraphs:
        if para.text.strip():
            full_text.append(para.text)
    
    for table in doc.tables:
        for rows in table.rows:
            row_text = [cell.text.strip() for cell in rows.cells]
            full_text.append(" | ".join(row_text))

    return "\n".join(full_text)
# ============================================================================
# ФУНКЦИЯ 11: УМНЫЙ ПОИСК ЦЕН
# ============================================================================
def search_with_tavily(query: str, max_results: int = 5) -> list:
    """
    Поиск через Tavily Search API.
    Возвращает список найденных цен/источников.
    
    Args:
        query: Поисковый запрос
        max_results: Максимум результатов
    
    Returns:
        list: Список словарей с рыночными данными
    """
    if not models.tavily_key:
        print("   ⚠️ TAVILY_API_KEY отсутствует")
        return []
    
    print(f"🔎 Tavily Search: {query}")
    
    try:
        tool = TavilySearchResults(
            max_results=max_results,
            search_depth="advanced",
            include_answer=True,
            include_raw_content=True,
            api_key=models.tavily_key
        )
        
        # Выполняем поиск
        results = tool.invoke(query)
        
        # Парсим результаты
        market_data = parse_tavily_results(results)
        
        print(f"   ✅ Tavily нашёл {len(market_data)} источников")
        return market_data
    
    except Exception as e:
        print(f"   ⚠️ Tavily ошибка: {e}")
        return []
    
# ============================================================================
# ФУНКЦИЯ 12: АДАПТИВНЫЙ ПОИСК ЦЕН (УМНАЯ ЛОГИКА)
# ============================================================================
def search_prices_adaptive(fact_sheet: dict) -> list:
    """
    Умный поиск цен в зависимости от типа закупки.
    ИСПРАВЛЕНО: Теперь ищет конкретный товар из tech_specs, а не общее название.
    """
    print("🔍 Адаптивный поиск цен...")
    
    # Извлекаем данные
    predmet_zakupki = fact_sheet.get("meta_info", {}).get("predmet_zakupki", "")
    tech_specs = fact_sheet.get("tech_specs", [])
    
    # ✅ ИСПРАВЛЕНИЕ: Определяем тип и конкретный товар/услугу
    procurement_type = "UNKNOWN"
    specific_item = predmet_zakupki  # Fallback
    
    if tech_specs:
        procurement_type = tech_specs[0].get("type", "UNKNOWN")
        
        # Ищем самый конкретный товар в спецификациях
        for spec in tech_specs:
            param = spec.get("parameter", "")
            value = spec.get("value", "")
            
            # Если есть конкретное название товара/модели
            if value and len(value) > 10 and not value.startswith("Не указано"):
                specific_item = value
                break
    
    print(f"   📦 Тип закупки: {procurement_type}")
    print(f"   🛒 Предмет: {specific_item}")
    
    market_data = []
    
    # === СТРАТЕГИЯ 1: ТОВАРЫ (GOODS) ===
    if procurement_type == "GOODS":
        print("  → Использую SerpAPI (Google Shopping)")
        
        # ✅ Улучшенный запрос с конкретным товаром
        search_query = f"{specific_item} цена Казахстан"
        
        try:
            market_data = search_market_prices(search_query)
            
            # Если SerpAPI не дал результатов - пробуем Tavily
            if not market_data and models.tavily_key:
                print("  → Fallback: Пробую Tavily Search...")
                tavily_results = search_with_tavily(search_query)
                market_data.extend(tavily_results)
                
        except Exception as e:
            print(f"   ⚠️ SerpAPI ошибка: {e}")
    
    # === СТРАТЕГИЯ 2: УСЛУГИ (SERVICES) ===
    elif procurement_type == "SERVICES":
        print("  → Использую Historical Database + Tavily")
        
        try:
            # 1. Поиск в исторических данных
            similar = check_similarity_risk(specific_item)
            
            if similar:
                for match in similar[:3]:
                    market_data.append({
                        "source": "Историческая база (36K контрактов)",
                        "title": f"Контракт #{match['past_contract']}",
                        "price": f"{match['past_amount']:,.0f} ₸",
                        "similarity": f"{match['similarity']}%"
                    })
                
                print(f"   ✅ Найдено {len(market_data)} похожих услуг")
            
            # 2. Дополняем Tavily
            if models.tavily_key:
                print("  → Дополнительный поиск через Tavily...")
                tavily_query = f"{specific_item} стоимость услуг Казахстан 2025"
                tavily_results = search_with_tavily(tavily_query, max_results=3)
                market_data.extend(tavily_results)
        
        except Exception as e:
            print(f"   ⚠️ Ошибка: {e}")
    
    # === СТРАТЕГИЯ 3: СТРОИТЕЛЬСТВО (CONSTRUCTION) ===
    elif procurement_type == "CONSTRUCTION":
        print("  → Использую Historical Database (строительные работы)")
        
        try:
            similar = check_similarity_risk(specific_item)
            if similar:
                for match in similar[:5]:
                    market_data.append({
                        "source": "Историческая база (работы)",
                        "title": f"Контракт #{match['past_contract']}",
                        "price": f"{match['past_amount']:,.0f} ₸"
                    })
        except Exception as e:
            print(f"   ⚠️ Ошибка: {e}")
    
    # === СТРАТЕГИЯ 4: НЕИЗВЕСТНО (UNKNOWN) ===
    else:
        print("  → Тип неизвестен, пробую все методы")
        
        # Сначала SerpAPI
        try:
            market_data = search_market_prices(f"{specific_item} цена")
        except:
            pass
        
        # Если пусто - Historical
        if not market_data:
            try:
                similar = check_similarity_risk(specific_item)
                if similar:
                    market_data.append({
                        "source": "Историческая база",
                        "price": f"{similar[0]['past_amount']:,.0f} ₸"
                    })
            except:
                pass
        
        if not market_data and models.tavily_key:
            tavily_results = search_with_tavily(f"{specific_item} цена Казахстан")
            market_data.extend(tavily_results)
    
    # === ФОЛЛБЭК: ЕСЛИ НИЧЕГО НЕ НАШЛИ ===
    if not market_data:
        print("   ⚠️ Цены не найдены ни одним методом")
        market_data.append({
            "source": "Оценка отсутствует",
            "title": "Требуется ручная проверка",
            "price": "Данных нет"
        })
    
    return market_data


# ============================================================================
# ФУНКЦИЯ 13: GPT-RESEARCHER ФИНАЛЬНЫЙ АНАЛИТИК
# ============================================================================

async def gpt_researcher_final_verdict(
    fact_sheets: list,
    qdrant_matches_list: list,
    market_data_list: list
) -> list:
    """
    🔥 ГИБРИДНЫЙ GPT-Researcher с автоматическим фоллбэком.
    
    Стратегия:
    1. Пытаемся Gemini 2.0 Flash (быстрый и дешевый)
    2. Если SAFETY блок или невалидный JSON → GPT-4o-mini
    3. Если GPT тоже падает → структурированная ошибка
    """
    print("\n" + "="*60)
    print("GPT-RESEARCHER: ФИНАЛЬНЫЙ АНАЛИЗ (HYBRID MODE)")
    print("="*60)
    
    verdicts = []
    
    for idx, fact_sheet in enumerate(fact_sheets):
        print(f"\n📋 Анализ тендера {idx+1}/{len(fact_sheets)}...")
        
        qdrant_matches = qdrant_matches_list[idx] if idx < len(qdrant_matches_list) else []
        market_data = market_data_list[idx] if idx < len(market_data_list) else []
        
        # Проверка качества данных
        data_quality = assess_data_quality(fact_sheet, market_data, qdrant_matches)
        print(f"   📊 Качество данных: {data_quality['score']}/100")
        
        # Если данных мало - делаем доп. поиск
        if data_quality['score'] < 50 and models.tavily_key:
            print("   🔍 Данных недостаточно! Запускаю доп. поиск...")
            item_name = fact_sheet.get("meta_info", {}).get("predmet_zakupki", "")
            if item_name:
                extra_search = search_with_tavily(
                    f"{item_name} рыночная цена анализ Казахстан", 
                    max_results=5
                )
                market_data.extend(extra_search)
                print(f"   ✅ Добавлено {len(extra_search)} источников")
        
        # ✅ ПОПЫТКА 1: GEMINI
        verdict = await try_gemini_analysis(
            fact_sheet, 
            market_data, 
            qdrant_matches, 
            data_quality
        )
        
        # ✅ ПОПЫТКА 2: GPT-4o-mini (если Gemini не сработал)
        if verdict.get("error"):
            print(f"   ⚠️ Gemini не смог: {verdict['error'][:100]}")
            print(f"   🔄 Переключаюсь на GPT-4o-mini...")
            
            verdict = await try_gpt_analysis(
                fact_sheet, 
                market_data, 
                qdrant_matches, 
                data_quality
            )
        
        verdicts.append(verdict)
        
        # Логирование
        if "error" not in verdict:
            risk_score = verdict.get("risk_score", 0)
            overall = verdict.get("overall_verdict", "UNKNOWN")
            model_used = verdict.get("model_used", "unknown")
            print(f"   ✅ Вердикт: {overall} (Risk: {risk_score}/100) [{model_used}]")
        else:
            print(f"   ❌ Финальная ошибка: {verdict['error'][:100]}")
    
    print("\n" + "="*60)
    print(f"✅ GPT-RESEARCHER ЗАВЕРШИЛ АНАЛИЗ: {len(verdicts)} вердиктов")
    print("="*60 + "\n")
    
    return verdicts


async def try_gemini_analysis(
    fact_sheet: dict,
    market_data: list,
    qdrant_matches: list,
    data_quality: dict
) -> dict:
    """
    Пытается выполнить анализ через Gemini.
    
    Returns:
        dict: Вердикт или {"error": "..."} при неудаче
    """
    try:
        # Формируем промпт
        research_prompt = build_research_prompt(
            fact_sheet, 
            market_data, 
            qdrant_matches, 
            data_quality
        )
        
        # ✅ Создаем модель с ОТКЛЮЧЕННЫМИ фильтрами безопасности
        model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            system_instruction="You are a data analyst. Analyze procurement data objectively. Respond ONLY with valid JSON. Never use line breaks inside string values.",
            generation_config={
                "temperature": 0.1,
                "max_output_tokens": 4000,
                "response_mime_type": "application/json"
            },
            # ✅ КРИТИЧНО: Отключаем SAFETY фильтры
            safety_settings={
                "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
                "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
                "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
                "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE"
            }
        )
        
        response = model.generate_content(research_prompt)
        
        # ✅ Проверяем finish_reason
        if not response.candidates:
            return {"error": "No candidates returned from Gemini"}
        
        candidate = response.candidates[0]
        finish_reason = candidate.finish_reason
        
        # finish_reason: 1=STOP (ok), 2=SAFETY, 3=MAX_TOKENS, 4=RECITATION
        if finish_reason == 2:
            return {"error": "SAFETY: Gemini blocked response due to safety filters"}
        
        if finish_reason == 3:
            return {"error": "MAX_TOKENS: Response too long"}
        
        if finish_reason == 4:
            return {"error": "RECITATION: Potential copyright issue"}
        
        # ✅ Безопасно извлекаем текст
        if not candidate.content or not candidate.content.parts:
            return {"error": "Empty response from Gemini"}
        
        result_text = candidate.content.parts[0].text
        
        # ✅ Очищаем JSON от возможных артефактов
        result_text = result_text.strip()
        
        # Удаляем markdown code blocks если есть
        if result_text.startswith("```json"):
            result_text = result_text[7:]
        if result_text.startswith("```"):
            result_text = result_text[3:]
        if result_text.endswith("```"):
            result_text = result_text[:-3]
        
        result_text = result_text.strip()
        
        # ✅ Пытаемся распарсить JSON
        try:
            verdict = json.loads(result_text)
        except json.JSONDecodeError as e:
            # Пытаемся восстановить
            print(f"   ⚠️ JSON parse error: {e}")
            print(f"   📄 Первые 200 символов ответа: {result_text[:200]}")
            
            # Пробуем fix
            import re
            result_text = re.sub(r'([^\\])\n', r'\1\\n', result_text)
            
            try:
                verdict = json.loads(result_text)
            except:
                return {"error": f"Invalid JSON from Gemini: {str(e)}"}
        
        # ✅ Добавляем метаданные
        verdict["model_used"] = "Gemini 2.0 Flash"
        verdict["data_quality_score"] = data_quality["score"]
        
        return verdict
    
    except Exception as e:
        return {"error": f"Gemini exception: {str(e)}"}


async def try_gpt_analysis(
    fact_sheet: dict,
    market_data: list,
    qdrant_matches: list,
    data_quality: dict
) -> dict:
    """
    Фоллбэк: анализ через GPT-4o-mini.
    
    Returns:
        dict: Вердикт или {"error": "..."} при неудаче
    """
    if not models.openai_client:
        return {
            "error": "OpenAI client not initialized",
            "tender_id": fact_sheet.get("lot_id", "unknown")
        }
    
    try:
        # Формируем промпт (тот же самый)
        research_prompt = build_research_prompt(
            fact_sheet, 
            market_data, 
            qdrant_matches, 
            data_quality
        )
        
        # ✅ Вызываем GPT-4o-mini
        response = models.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system", 
                    "content": "You are a procurement auditor. Analyze data objectively and respond ONLY with valid JSON."
                },
                {
                    "role": "user", 
                    "content": research_prompt
                }
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=4000
        )
        
        result_text = response.choices[0].message.content
        
        # Парсим JSON
        verdict = json.loads(result_text)
        
        # Добавляем метаданные
        verdict["model_used"] = "GPT-4o-mini"
        verdict["data_quality_score"] = data_quality["score"]
        
        return verdict
    
    except Exception as e:
        return {
            "error": f"GPT exception: {str(e)}",
            "tender_id": fact_sheet.get("lot_id", "unknown")
        }


def build_research_prompt(
    fact_sheet: dict,
    market_data: list,
    qdrant_matches: list,
    data_quality: dict
) -> str:
    """
    Строит универсальный промпт для обеих моделей.
    """
    return f"""
You are an elite procurement auditor analyzing government tenders for corruption risks.
####ANSWER IN RUSSIAN
# DATA FOR ANALYSIS:

## 1. TENDER FACTS:
{json.dumps(fact_sheet, ensure_ascii=False, indent=2)}

## 2. MARKET PRICES:
{json.dumps(market_data, ensure_ascii=False, indent=2)}

## 3. HISTORICAL RISKS:
{json.dumps(qdrant_matches, ensure_ascii=False, indent=2)}

## 4. DATA QUALITY:
- Completeness: {data_quality['score']}/100
- Price sources: {len(market_data)}
- Historical matches: {len(qdrant_matches)}

# ANALYSIS INSTRUCTIONS:

**STEP 1: Price Analysis**
- Compare `budget_total` with market prices
- If deviation >15% → RISK
- If no data → REQUIRES_VERIFICATION

**STEP 2: Historical Check**
- If similarity >85% → HIGH_RISK (scheme copy)
- If buyer matches → ATTENTION

**STEP 3: Technical Analysis**
- Check tech_specs for inflated requirements
- Check legal_requirements for competition barriers

**STEP 4: Final Verdict**
- Consider ALL factors
- Be objective but strict

# RESPONSE FORMAT (STRICT JSON):

CRITICAL: 
1. All text fields must be single line
2. Use \\n for line breaks inside strings
3. Validate JSON before sending

{{
  "tender_id": "File name or ID",
  "tender_summary": {{
    "product": "Brief description",
    "declared_price": 1000000,
    "market_price_avg": 800000,
    "procurement_type": "GOODS/SERVICES/CONSTRUCTION"
  }},
  "risk_factors": [
    {{
      "type": "PRICE_OVERESTIMATION | HISTORICAL_PATTERN | TECH_BARRIER | SPLIT_PROCUREMENT",
      "severity": "LOW | MEDIUM | HIGH | CRITICAL",
      "description": "Specific risk description with numbers IN ONE LINE",
      "evidence": "Source reference or contract number"
    }}
  ],
  "data_sources_used": {{
    "serp_api": {len([m for m in market_data if 'Google' in str(m.get('source', ''))])},
    "tavily_search": {len([m for m in market_data if 'http' in str(m.get('source', ''))])},
    "historical_db": {len(qdrant_matches)}
  }},
  "overall_verdict": "SAFE | REQUIRES_VERIFICATION | HIGH_RISK | CRITICAL_RISK",
  "risk_score": 0-100,
  "conclusion_markdown": "# Detailed conclusion\\n\\nDetailed analysis IN ONE LINE with \\n for breaks..."
}}

Be objective. If no risks - say so. But if risks exist - state clearly with evidence. 
"""


# ============================================================================
# ФУНКЦИЯ 14: СОХРАНЕНИЕ ПОДТВЕРЖДЕННОГО РИСКА В БАЗУ
# ============================================================================
def save_approved_risk_to_qdrant(verdict: dict):
    """
    Сохраняет подтвержденный риск в резервную базу approved_risks.
    Вызывается только если пользователь подтвердил вердикт.
    """
    print(f"Сохранение вердикта в резервную базу...")
    
    try:
        # Формируем текст для векторизации
        product = verdict.get("tender_summary", {}).get("product", "")
        risks = verdict.get("risk_factors", [])
        conclusion = verdict.get("conclusion_markdown", "")
        
        text_for_vector = (
            f"passage: Товар/услуга: {product}. "
            f"Риски: {', '.join([r.get('description', '') for r in risks])}. "
            f"Заключение: {conclusion[:500]}"
        )
        
        # Векторизуем
        vector = models.embedding_model.encode(text_for_vector).tolist()
        
        # Создаем точку
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload=verdict  # Сохраняем весь вердикт
        )
        
        # Загружаем в резервную коллекцию
        models.qdrant_client.upsert(
            collection_name=models.collection_name_v2,
            points=[point]
        )
        
        print(f" Вердикт сохранен в '{models.collection_name_v2}'")
        return True
    
    except Exception as e:
        print(f" Ошибка сохранения: {e}")
        return False


# ============================================================================
# ФУНКЦИЯ 15: BATCH ОБРАБОТКА (ГЛАВНАЯ ОРКЕСТРАЦИЯ)
# ============================================================================
async def process_multiple_tenders_batch(search_queries: list) -> dict:
    """
    Обрабатывает несколько тендеров батчем.
    
    Args:
        search_queries: Список поисковых запросов ["серверы Dell", "автобусы школьные"]
    
    Returns:
        dict: {
            "fact_sheets": [...],
            "qdrant_matches": [...],
            "market_data": [...],
            "final_verdicts": [...]
        }
    """
    print("\n" + "-"*30)
    print(f"BATCH ОБРАБОТКА: {len(search_queries)} запросов")
    print("-"*30 + "\n")
    
    fact_sheets = []
    qdrant_matches_list = []
    market_data_list = []
    
    # СБОР ДАННЫХ ПО ВСЕМ ЗАПРОСАМ ===
    for idx, query in enumerate(search_queries):
        print(f"\n{'='*60}")
        print(f"ЗАПРОС {idx+1}/{len(search_queries)}: {query}")
        print(f"{'='*60}\n")
        
        # 1. Скачать и распарсить
        tender_objects = search_download_and_parse(query, target_count=1)
        
        if not tender_objects:
            print(f"   ⚠️ Документы не найдены для '{query}'")
            fact_sheets.append({"error": "Не найдено", "query": query})
            qdrant_matches_list.append([])
            market_data_list.append([])
            continue
        
        # Берем первый объект
        tender_obj = tender_objects[0]
        
        # Извлечь факты
        fact_sheet = extract_facts_for_researcher(tender_obj)
        fact_sheets.append(fact_sheet)
        
        # Проверить историю рисков
        item_name = fact_sheet.get("meta_info", {}).get("predmet_zakupki", query)
        qdrant_matches = check_similarity_risk(item_name)
        qdrant_matches_list.append(qdrant_matches or [])
        
        # Найти рыночные цены (адаптивно)
        market_data = search_prices_adaptive(fact_sheet)
        market_data_list.append(market_data)
    
    # GPT-RESEARCHER ФИНАЛЬНЫЙ АНАЛИЗ ===
    final_verdicts = await gpt_researcher_final_verdict(
        fact_sheets=fact_sheets,
        qdrant_matches_list=qdrant_matches_list,
        market_data_list=market_data_list
    )
    
    # ВОЗВРАТ РЕЗУЛЬТАТОВ
    return {
        "queries": search_queries,
        "fact_sheets": fact_sheets,
        "qdrant_matches": qdrant_matches_list,
        "market_data": market_data_list,
        "final_verdicts": final_verdicts
    }
