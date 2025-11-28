import re
import json


def extract_price_from_text(text: str) -> float:
    """
    Извлекает цену из текста.
    
    Примеры:
    - "₸500,000" - 500000.0
    - "$1,234.56" - 1234.56
    - "123 456 руб" - 123456.0
    """
    if not text:
        return 0.0
    
    cleaned = re.sub(r'[^\d.,]', '', text)
    
    cleaned = cleaned.replace(',', '')
    
    try:
        return float(cleaned)
    except:
        return 0.0


def assess_data_quality(fact_sheet: dict, market_data: list, qdrant_matches: list) -> dict:
    """
    Оценивает качество собранных данных.
    
    Returns:
        dict: {
            "score": int (0-100),
            "reasons": list[str]
        }
    """
    score = 0
    reasons = []
    
    # 1. Проверка факт-листа (40 баллов)
    if fact_sheet and "error" not in fact_sheet:
        score += 20
        
        # Есть бюджет?
        budget = fact_sheet.get("finance", {}).get("budget_total", 0)
        if budget and budget > 0:
            score += 10
        else:
            reasons.append("Бюджет не указан")
        
        # Есть tech_specs?
        tech_specs = fact_sheet.get("tech_specs", [])
        if tech_specs and len(tech_specs) > 0:
            score += 10
        else:
            reasons.append("Технические характеристики отсутствуют")
    else:
        reasons.append("Ошибка извлечения фактов")
    
    # 2. Проверка рыночных данных (40 баллов)
    if market_data:
        score += min(len(market_data) * 8, 40)  # До 40 баллов (5 источников)
    else:
        reasons.append("Рыночные цены не найдены")
    
    # 3. Проверка исторических данных (20 баллов)
    if qdrant_matches:
        score += min(len(qdrant_matches) * 4, 20)  # До 20 баллов (5 совпадений)
    
    return {
        "score": min(score, 100),
        "reasons": reasons if reasons else ["Все данные в норме"]
    }


def parse_tavily_results(results: list) -> list:
    """
    Парсит результаты Tavily Search в унифицированный формат.
    
    Args:
        results: Список результатов от TavilySearchResults
    
    Returns:
        list: [{"source": str, "title": str, "price": str, "content": str}, ...]
    """
    parsed = []
    
    for result in results:
        if isinstance(result, dict):
            # Извлекаем цену из контента
            content = result.get("content", "")
            url = result.get("url", "")
            title = result.get("title", "")
            
            # Пытаемся найти цену в контенте
            price_match = re.search(r'(\d[\d\s,]*\.?\d*)\s*(₸|тенге|KZT|руб|USD|\$)', content, re.IGNORECASE)
            price = price_match.group(0) if price_match else "Цена не указана"
            
            parsed.append({
                "source": url,
                "title": title or "Без названия",
                "price": price,
                "content": content[:200] + "..." if len(content) > 200 else content
            })
    
    return parsed


def format_verdict_for_display(verdict: dict) -> str:
    """
    Форматирует вердикт для красивого вывода в консоль.
    
    Args:
        verdict: Словарь с вердиктом от GPT-Researcher
    
    Returns:
        str: Отформатированный текст для вывода
    """
    if "error" in verdict:
        return f"""
╔══════════════════════════════════════════════════════════════╗
║                       ❌ ОШИБКА АНАЛИЗА                      ║
╚══════════════════════════════════════════════════════════════╝
Тендер: {verdict.get('tender_id', 'unknown')}
Ошибка: {verdict.get('error', 'Unknown error')}
"""
    
    # Извлекаем данные
    tender_id = verdict.get("tender_id", "N/A")
    summary = verdict.get("tender_summary", {})
    risk_score = verdict.get("risk_score", 0)
    overall = verdict.get("overall_verdict", "UNKNOWN")
    risks = verdict.get("risk_factors", [])
    conclusion = verdict.get("conclusion_markdown", "")
    
    # Определяем эмодзи по вердикту
    emoji_map = {
        "БЕЗОПАСНО": "✅",
        "ТРЕБУЕТ_ПРОВЕРКИ": "⚠️",
        "ВЫСОКИЙ_РИСК": "🚨",
        "КРИТИЧЕСКИЙ_РИСК": "💀"
    }
    emoji = emoji_map.get(overall, "❓")
    
    # Форматируем вывод
    output = f"""
╔══════════════════════════════════════════════════════════════╗
║                    {emoji} ВЕРДИКТ: {overall}                    ║
╚══════════════════════════════════════════════════════════════╝

📋 ТЕНДЕР: {tender_id}
📦 ПРЕДМЕТ: {summary.get('product', 'N/A')}
💰 ЗАЯВЛЕННАЯ ЦЕНА: {summary.get('declared_price', 0):,.0f} ₸
💵 РЫНОЧНАЯ ЦЕНА: {summary.get('market_price_avg', 0):,.0f} ₸
📊 РИСК-СКОР: {risk_score}/100

"""
    
    # Выводим риски
    if risks:
        output += "⚠️  ОБНАРУЖЕННЫЕ РИСКИ:\n"
        output += "─" * 60 + "\n"
        
        for idx, risk in enumerate(risks, 1):
            risk_type = risk.get('type', 'UNKNOWN')
            severity = risk.get('severity', 'UNKNOWN')
            description = risk.get('description', 'N/A')
            evidence = risk.get('evidence', 'N/A')
            
            # Эмодзи для severity
            severity_emoji = {
                "LOW": "🟢",
                "MEDIUM": "🟡",
                "HIGH": "🟠",
                "CRITICAL": "🔴"
            }
            s_emoji = severity_emoji.get(severity, "⚪")
            
            output += f"""
{idx}. {s_emoji} {risk_type} [{severity}]
   Описание: {description}
   Доказательства: {evidence}
"""
    else:
        output += "✅ Риски не обнаружены\n"
    
    # Добавляем заключение (если есть)
    if conclusion:
        output += "\n" + "─" * 60 + "\n"
        output += "📝 ЗАКЛЮЧЕНИЕ:\n"
        
        # Конвертируем Markdown в простой текст для консоли
        conclusion_text = conclusion.replace("\\n", "\n").replace("#", "")
        output += conclusion_text[:500]  # Ограничиваем длину
        
        if len(conclusion_text) > 500:
            output += "\n... (полный текст в JSON файле)"
    
    output += "\n" + "═" * 60 + "\n"
    
    return output


def convert_currency_to_kzt(amount: float, currency: str) -> float:
    """
    Конвертирует валюту в тенге.
    
    Args:
        amount: Сумма
        currency: Валюта (KZT, USD, RUB, EUR)
    
    Returns:
        float: Сумма в тенге
    """
    rates = {
        "KZT": 1.0,
        "USD": 450.0,
        "RUB": 5.0,
        "EUR": 500.0,
        "$": 450.0,
        "₸": 1.0
    }
    
    currency_upper = currency.upper().strip()
    rate = rates.get(currency_upper, 1.0)
    
    return amount * rate


def clean_price_text(text: str) -> str:
    """
    Очищает текст цены от лишних символов.
    
    Примеры:
    - "от 100 000 ₸" → "100000"
    - "$1,234.56 USD" → "1234.56"
    """
    # Удаляем слова
    text = re.sub(r'(от|до|около|примерно|цена|стоимость)', '', text, flags=re.IGNORECASE)
    
    # Оставляем только цифры, точки и запятые
    cleaned = re.sub(r'[^\d.,]', '', text)
    
    return cleaned


def validate_verdict_json(verdict: dict) -> bool:
    """
    Проверяет, что вердикт содержит все необходимые поля.
    
    Args:
        verdict: Словарь с вердиктом
    
    Returns:
        bool: True если валидный
    """
    required_fields = [
        "tender_id",
        "tender_summary",
        "risk_factors",
        "overall_verdict",
        "risk_score"
    ]
    
    for field in required_fields:
        if field not in verdict:
            return False
    
    # Проверка tender_summary
    summary = verdict.get("tender_summary", {})
    if not isinstance(summary, dict):
        return False
    
    # Проверка risk_factors
    risks = verdict.get("risk_factors", [])
    if not isinstance(risks, list):
        return False
    
    return True