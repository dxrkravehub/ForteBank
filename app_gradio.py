import gradio as gr
import asyncio
import json
import os
from datetime import datetime
from typing import List, Dict
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Backend для серверного рендеринга
import seaborn as sns
import numpy as np

# Импорты из твоей системы
from main import run_agent_batch, initialize_qdrant_database, models
from agent_functions import (
    search_download_and_parse,
    extract_facts_for_researcher,
    check_similarity_risk,
    search_prices_adaptive,
    gpt_researcher_final_verdict,
    save_approved_risk_to_qdrant,
    extract_data_from_pdf,
    docx_processing
)
from cache_utils import get_cache_stats, clear_cache, is_redis_available
from utils import format_verdict_for_display

# Глобальные настройки моделей
AVAILABLE_MODELS = {
    "Gemini 2.5 Flash": "gemini-2.5-flash",
    "GPT-4o-mini": "gpt-4o-mini",
    "Claude Sonnet 3.5": "claude-sonnet-3.5"  # Если подключишь
}

CURRENT_MODEL = "gemini-2.5-flash"

# Путь к CSV для визуализации
CSV_PATH = "goszakup_contracts_FINAL_ANALYSIS.csv"  # Замени на свой путь

# Кастомный CSS для финтех стиля
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

:root {
    --primary-bg: #0A0E27;
    --secondary-bg: #141B34;
    --card-bg: #1A2238;
    --accent-color: #00D9FF;
    --accent-hover: #00B8D4;
    --success-color: #00FF9C;
    --warning-color: #FFB800;
    --danger-color: #FF3B5C;
    --text-primary: #FFFFFF;
    --text-secondary: #B4C1D9;
    --border-color: #2A3F5F;
}

body, .gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background: linear-gradient(135deg, var(--primary-bg) 0%, #0F1419 100%) !important;
}

/* HERO SECTION - Приветствие */
.hero-container {
    text-align: center;
    padding: 60px 20px 40px 20px;
    background: linear-gradient(180deg, rgba(0, 217, 255, 0.05) 0%, transparent 100%);
    border-radius: 20px;
    margin-bottom: 40px;
}

.hero-title {
    font-size: 4em !important;
    font-weight: 900 !important;
    background: linear-gradient(135deg, #00D9FF 0%, #00FF9C 50%, #FFB800 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 16px !important;
    letter-spacing: -0.03em;
    line-height: 1.2;
    animation: gradient-shift 3s ease infinite;
    background-size: 200% 200%;
}

@keyframes gradient-shift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

.hero-subtitle {
    font-size: 1.3em !important;
    color: var(--text-secondary) !important;
    font-weight: 400 !important;
    margin-top: 0 !important;
    opacity: 0.9;
}

/* Главный контейнер */
.main-container {
    background: var(--secondary-bg) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 16px !important;
    padding: 24px !important;
    box-shadow: 0 8px 32px rgba(0, 217, 255, 0.1) !important;
}

/* Заголовки */
h1, h2, h3 {
    color: var(--text-primary) !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em !important;
}

.logo-title {
    background: linear-gradient(135deg, var(--accent-color) 0%, #00FF9C 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.5em !important;
    font-weight: 800 !important;
    text-align: center;
    margin-bottom: 8px;
}

.subtitle {
    color: var(--text-secondary) !important;
    text-align: center;
    font-size: 1.1em;
    margin-bottom: 32px;
}

/* Карточки */
.card {
    background: var(--card-bg) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 12px !important;
    padding: 20px !important;
    margin: 12px 0 !important;
    transition: all 0.3s ease !important;
}

.card:hover {
    border-color: var(--accent-color) !important;
    box-shadow: 0 4px 16px rgba(0, 217, 255, 0.15) !important;
    transform: translateY(-2px);
}

/* Кнопки */
button.primary {
    background: linear-gradient(135deg, var(--accent-color) 0%, #0099CC 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 12px 24px !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 12px rgba(0, 217, 255, 0.3) !important;
}

button.primary:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(0, 217, 255, 0.4) !important;
}

button.secondary {
    background: var(--card-bg) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 8px !important;
    padding: 10px 20px !important;
    transition: all 0.3s ease !important;
}

button.secondary:hover {
    border-color: var(--accent-color) !important;
    background: var(--secondary-bg) !important;
}

/* Инпуты */
input, textarea, select {
    background: var(--secondary-bg) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
    padding: 12px !important;
    transition: all 0.3s ease !important;
}

input:focus, textarea:focus, select:focus {
    border-color: var(--accent-color) !important;
    box-shadow: 0 0 0 3px rgba(0, 217, 255, 0.1) !important;
    outline: none !important;
}

/* Метрики */
.metric-card {
    background: linear-gradient(135deg, var(--card-bg) 0%, #1F2937 100%) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 12px !important;
    padding: 20px !important;
    text-align: center !important;
}

.metric-value {
    font-size: 2.5em !important;
    font-weight: 700 !important;
    background: linear-gradient(135deg, var(--accent-color) 0%, #00FF9C 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.metric-label {
    color: var(--text-secondary) !important;
    font-size: 0.9em !important;
    margin-top: 8px !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Статусы */
.status-safe {
    color: var(--success-color) !important;
    font-weight: 600 !important;
}

.status-warning {
    color: var(--warning-color) !important;
    font-weight: 600 !important;
}

.status-danger {
    color: var(--danger-color) !important;
    font-weight: 600 !important;
}

/* Прогресс бар */
.progress-bar {
    background: var(--secondary-bg) !important;
    height: 8px !important;
    border-radius: 4px !important;
    overflow: hidden !important;
}

.progress-fill {
    background: linear-gradient(90deg, var(--accent-color) 0%, var(--success-color) 100%) !important;
    height: 100% !important;
    transition: width 0.5s ease !important;
}

/* Сайдбар */
.sidebar {
    background: var(--card-bg) !important;
    border-right: 1px solid var(--border-color) !important;
    padding: 24px !important;
}

/* Анимация загрузки */
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}

.loading {
    animation: pulse 2s ease-in-out infinite;
}

/* Badges */
.badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 12px;
    font-size: 0.85em;
    font-weight: 600;
    margin: 4px;
}

.badge-cached {
    background: rgba(0, 255, 156, 0.15);
    color: var(--success-color);
    border: 1px solid var(--success-color);
}

.badge-new {
    background: rgba(0, 217, 255, 0.15);
    color: var(--accent-color);
    border: 1px solid var(--accent-color);
}

.badge-risk {
    background: rgba(255, 59, 92, 0.15);
    color: var(--danger-color);
    border: 1px solid var(--danger-color);
}

/* Таблицы */
table {
    background: var(--card-bg) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 8px !important;
}

th {
    background: var(--secondary-bg) !important;
    color: var(--text-primary) !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    font-size: 0.85em;
    letter-spacing: 0.05em;
}

td {
    color: var(--text-secondary) !important;
    border-color: var(--border-color) !important;
}

tr:hover {
    background: var(--secondary-bg) !important;
}

/* Графики */
.plot-container {
    background: var(--card-bg) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 12px !important;
    padding: 20px !important;
    margin: 20px 0 !important;
}
"""

async def process_queries_async(queries_text: str, use_cache: bool, model_choice: str):
    """Асинхронная обработка запросов"""
    if not queries_text.strip():
        return "⚠️ Введите хотя бы один запрос!", None, None
    
    queries = [q.strip() for q in queries_text.split("\n") if q.strip()]
    
    # Обновляем модель если нужно
    global CURRENT_MODEL
    CURRENT_MODEL = AVAILABLE_MODELS.get(model_choice, "gemini-2.5-flash")
    
    # Очистка кеша если нужно
    if not use_cache:
        clear_cache("procurement:batch_verdict:*")
    
    # Запуск агента
    try:
        result = await run_agent_batch(queries)
        
        if not result or "final_state" not in result:
            return "❌ Ошибка: агент не вернул результат", None, None
        
        final_state = result["final_state"]
        verdicts = final_state.get("final_verdicts", [])
        cache_stats = final_state.get("cache_stats", {})
        
        # Формируем отчёт
        report = format_results_report(verdicts, cache_stats, queries)
        
        # Формируем DataFrame для таблицы
        df = create_verdicts_dataframe(verdicts)
        
        # Формируем детальный JSON
        details = json.dumps(final_state, ensure_ascii=False, indent=2)
        
        return report, df, details
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return f"❌ Ошибка выполнения:\n{str(e)}\n\n{error_details}", None, None


def format_results_report(verdicts: List[Dict], cache_stats: Dict, queries: List[str]) -> str:
    """Форматирует результаты в красивый отчёт"""
    report = f"""
# 📊 Результаты анализа
**Обработано запросов:** {len(queries)}
**Получено вердиктов:** {len(verdicts)}

---

## 🎯 Статистика кеша
- **Cache Hit Rate:** {cache_stats.get('cache_hit_rate', 0):.1f}%
- **Попадания:** {cache_stats.get('cache_hits', 0)}
- **Промахи:** {cache_stats.get('cache_misses', 0)}
- **Новых закешировано:** {cache_stats.get('new_cached', 0)}

---

## 🔍 Детали по тендерам

"""
    
    for idx, verdict in enumerate(verdicts, 1):
        if "error" in verdict:
            report += f"\n### ❌ Тендер {idx}: ОШИБКА\n{verdict['error']}\n"
            continue
        
        # Извлекаем данные
        tender_summary = verdict.get("tender_summary", {})
        product = tender_summary.get("product", "Неизвестно")
        declared_price = tender_summary.get("declared_price", 0)
        market_price = tender_summary.get("market_price_avg", 0)
        overall_verdict = verdict.get("overall_verdict", "UNKNOWN")
        risk_score = verdict.get("risk_score", 0)
        from_cache = verdict.get("_from_cache", False)
        
        # Определяем эмодзи по вердикту
        if overall_verdict == "SAFE":
            emoji = "✅"
            status_class = "status-safe"
        elif overall_verdict == "REQUIRES_VERIFICATION":
            emoji = "⚠️"
            status_class = "status-warning"
        else:
            emoji = "🚨"
            status_class = "status-danger"
        
        cache_badge = "🎯 [CACHED]" if from_cache else "🆕 [NEW]"
        
        report += f"""
### {emoji} Тендер {idx}: {product[:50]}...
{cache_badge}

**Статус:** <span class="{status_class}">{overall_verdict}</span>  
**Риск-скор:** {risk_score}/100  
**Заявленная цена:** {declared_price:,.0f} ₸  
**Рыночная цена:** {market_price:,.0f} ₸  
**Отклонение:** {((declared_price - market_price) / market_price * 100) if market_price > 0 else 0:.1f}%

"""
        
        # Риски
        risk_factors = verdict.get("risk_factors", [])
        if risk_factors:
            report += "**Обнаруженные риски:**\n"
            for risk in risk_factors:
                severity = risk.get("severity", "UNKNOWN")
                risk_type = risk.get("type", "")
                description = risk.get("description", "")
                
                if severity == "CRITICAL":
                    risk_emoji = "🔴"
                elif severity == "HIGH":
                    risk_emoji = "🟠"
                elif severity == "MEDIUM":
                    risk_emoji = "🟡"
                else:
                    risk_emoji = "🟢"
                
                report += f"- {risk_emoji} **{risk_type}**: {description}\n"
        
        report += "\n---\n"
    
    return report


def create_verdicts_dataframe(verdicts: List[Dict]) -> pd.DataFrame:
    """Создаёт DataFrame из вердиктов для табличного отображения"""
    data = []
    
    for idx, verdict in enumerate(verdicts, 1):
        if "error" in verdict:
            data.append({
                "№": idx,
                "Продукт": "Ошибка",
                "Статус": "ERROR",
                "Риск": 0,
                "Цена тендера": 0,
                "Рыночная цена": 0,
                "Отклонение %": 0,
                "Кеш": "❌"
            })
            continue
        
        tender_summary = verdict.get("tender_summary", {})
        product = tender_summary.get("product", "Неизвестно")[:40]
        declared_price = tender_summary.get("declared_price", 0)
        market_price = tender_summary.get("market_price_avg", 0)
        overall_verdict = verdict.get("overall_verdict", "UNKNOWN")
        risk_score = verdict.get("risk_score", 0)
        from_cache = verdict.get("_from_cache", False)
        
        deviation = ((declared_price - market_price) / market_price * 100) if market_price > 0 else 0
        
        data.append({
            "№": idx,
            "Продукт": product,
            "Статус": overall_verdict,
            "Риск": risk_score,
            "Цена тендера": f"{declared_price:,.0f} ₸",
            "Рыночная цена": f"{market_price:,.0f} ₸",
            "Отклонение %": f"{deviation:.1f}%",
            "Кеш": "🎯" if from_cache else "🆕"
        })
    
    return pd.DataFrame(data)


def get_system_status():
    """Получает статус системы"""
    status = {
        "Redis": "🟢 Активен" if is_redis_available() else "🔴 Отключен",
        "Qdrant": "🟢 Подключен" if models.qdrant_client else "🔴 Отключен",
        "Gemini": "🟢 Доступен" if models.gemini_model else "🔴 Недоступен",
        "OpenAI": "🟢 Доступен" if models.openai_client else "🔴 Недоступен",
        "SerpAPI": "🟢 Подключен" if models.serp_api_key else "🔴 Отключен",
        "Tavily": "🟢 Подключен" if models.tavily_key else "🔴 Отключен"
    }
    
    cache_stats = get_cache_stats()
    
    status_text = "## 🔧 Статус системы\n\n"
    for service, state in status.items():
        status_text += f"**{service}:** {state}\n"
    
    status_text += f"\n## 📊 Статистика кеша\n\n"
    status_text += f"**Использование памяти:** {cache_stats.get('memory_used', 'N/A')}\n"
    status_text += f"**Ключей в кеше:** {cache_stats.get('total_keys', 0)}\n"
    
    return status_text


def create_ui():
    """Создаёт Gradio интерфейс"""
    
    with gr.Blocks(css=CUSTOM_CSS, theme=gr.themes.Base()) as app:
        gr.HTML("""
        <div class="hero-container">
            <h1 class="hero-title">I'm ForteGuard AI 🔍</h1>
            <p class="hero-subtitle">What can I help you today?</p>
        </div>
        """)
        # Заголовок
        gr.Markdown('<h1 class="logo-title">🛡️ СИСТЕМА АУДИТА ГОСЗАКУПОК</h1>')
        gr.Markdown('<p class="subtitle">Интеллектуальная система обнаружения коррупционных рисков</p>')
        
        with gr.Row():
            # Левая колонка - сайдбар с настройками
            with gr.Column(scale=1, elem_classes="sidebar"):
                gr.Markdown("### ⚙️ Настройки")
                
                model_selector = gr.Dropdown(
                    choices=list(AVAILABLE_MODELS.keys()),
                    value="Gemini 2.5 Flash",
                    label="🤖 Модель анализа",
                    info="Выберите AI модель для анализа"
                )
                
                use_cache_checkbox = gr.Checkbox(
                    value=True,
                    label="💾 Использовать кеш",
                    info="Ускоряет повторные запросы"
                )
                
                gr.Markdown("---")
                
                # Статус системы
                status_box = gr.Markdown(get_system_status())
                
                refresh_status_btn = gr.Button("🔄 Обновить статус", size="sm")
                refresh_status_btn.click(fn=get_system_status, outputs=status_box)
                
                gr.Markdown("---")
                
                clear_cache_btn = gr.Button("🗑️ Очистить кеш", size="sm")
                cache_status = gr.Markdown("")
                
                def clear_cache_action():
                    count = clear_cache("procurement:batch_verdict:*")
                    return f"✅ Очищено {count} записей"
                
                clear_cache_btn.click(fn=clear_cache_action, outputs=cache_status)
            
            # Правая колонка - основной интерфейс
            with gr.Column(scale=3):
                
                # Вкладки
                with gr.Tabs():
                    
                    # Вкладка 1: Анализ тендеров
                    with gr.Tab("🔍 Анализ тендеров"):
                        gr.Markdown("### Введите запросы для поиска (по одному на строку)")
                        
                        queries_input = gr.Textbox(
                            label="Поисковые запросы",
                            placeholder="Например:\nсерверы Dell PowerEdge\nавтобусы школьные Yutong\nмедицинское оборудование",
                            lines=5,
                            elem_classes="main-container"
                        )
                        
                        with gr.Row():
                            analyze_btn = gr.Button("🚀 Начать анализ", variant="primary", size="lg")
                            example_btn = gr.Button("📋 Загрузить пример", size="lg")
                        
                        # Функция для примера
                        def load_example():
                            return "серверы Dell PowerEdge\nавтобусы школьные Yutong\nмедицинское оборудование МРТ"
                        
                        example_btn.click(fn=load_example, outputs=queries_input)
                        
                        gr.Markdown("---")
                        
                        # Результаты
                        with gr.Row():
                            with gr.Column():
                                results_report = gr.Markdown(label="Отчёт")
                        
                        with gr.Row():
                            results_table = gr.Dataframe(
                                label="📊 Сводная таблица",
                                interactive=False
                            )
                        
                        with gr.Accordion("📄 Детальный JSON", open=False):
                            results_json = gr.Code(
                                label="Полные данные",
                                language="json"
                            )
                        
                        # Обработка нажатия кнопки анализа
                        def analyze_wrapper(queries, use_cache, model):
                            return asyncio.run(process_queries_async(queries, use_cache, model))
                        
                        analyze_btn.click(
                            fn=analyze_wrapper,
                            inputs=[queries_input, use_cache_checkbox, model_selector],
                            outputs=[results_report, results_table, results_json]
                        )
                    
                    # Вкладка 2: Статистика
                    with gr.Tab("📈 Статистика"):
                        gr.Markdown("### Аналитика работы системы")
                        
                        with gr.Row():
                            with gr.Column():
                                gr.Markdown("""
                                <div class="metric-card">
                                    <div class="metric-value" id="total-analyzed">0</div>
                                    <div class="metric-label">Проанализировано</div>
                                </div>
                                """)
                            
                            with gr.Column():
                                gr.Markdown("""
                                <div class="metric-card">
                                    <div class="metric-value" id="risks-found">0</div>
                                    <div class="metric-label">Рисков найдено</div>
                                </div>
                                """)
                            
                            with gr.Column():
                                gr.Markdown("""
                                <div class="metric-card">
                                    <div class="metric-value" id="cache-rate">0%</div>
                                    <div class="metric-label">Cache Hit Rate</div>
                                </div>
                                """)
                        
                        gr.Markdown("---")
                        
                        stats_details = gr.Markdown("""
                        ### 📊 Детальная статистика
                        
                        Здесь будет отображаться подробная информация о работе системы после выполнения анализа.
                        """)
                    
                    # Вкладка 3: О системе
                    with gr.Tab("ℹ️ О системе"):
                        gr.Markdown("""
                        # 🛡️ Система аудита госзакупок
                        
                        ## Возможности
                        
                        ### 🔍 Интеллектуальный анализ
                        - Автоматический поиск и загрузка документов тендеров
                        - Извлечение структурированных данных с помощью AI
                        - Сравнение с рыночными ценами через множество источников
                        
                        ### 🎯 Обнаружение рисков
                        - **Завышение цен**: Сравнение с рыночными аналогами
                        - **Дробление закупок**: Анализ паттернов разделения контрактов
                        - **Исторические схемы**: Векторный поиск похожих нарушений
                        - **Технические барьеры**: Выявление завышенных требований
                        
                        ### 🚀 Технологии
                        - **LangGraph**: Оркестрация агентов
                        - **Qdrant**: Векторная база для семантического поиска
                        - **Gemini 2.5 Flash**: Быстрый анализ документов
                        - **GPT-4o-mini**: Резервная модель
                        - **Redis**: Кеширование результатов
                        - **Playwright**: Автоматизация браузера
                        - **SerpAPI + Tavily**: Поиск рыночных данных
                        
                        ### 📊 Метрики качества
                        - Автоматическая оценка полноты данных
                        - Дополнительный поиск при низком качестве
                        - Гибридный режим (Gemini + GPT fallback)
                        
                        ---
                        **Автор:** Pied Piper Team
                        
                        **Версия:** 2.0  
                        **Последнее обновление:** 2025-01  
                        """)
        
        # Футер
        gr.Markdown("""
        ---
        <div style="text-align: center; color: #B4C1D9; font-size: 0.9em;">
            © 2025 Система аудита госзакупок • Разработано с использованием LangGraph & Gradio
        </div>
        """)
    
    return app


# Запуск
if __name__ == "__main__":
    print("🚀 Запуск Gradio UI...")
    
    # Инициализация базы данных (если нужно)
    # initialize_qdrant_database("path/to/your/csv")
    
    app = create_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True
    )