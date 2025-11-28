import nest_asyncio
import asyncio

nest_asyncio.apply()
import sys

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

import operator
import json
import asyncio
import os
import pandas as pd
from typing import TypedDict, List, Annotated, Literal
from IPython.display import Image, display


from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver

# Импортируем наши функции
from agent_functions import (
    models,
    LoadDFinQdrant,
    search_download_and_parse,
    extract_facts_for_researcher,
    check_similarity_risk,
    search_prices_adaptive,
    gpt_researcher_final_verdict,
    save_approved_risk_to_qdrant,
    search_with_tavily,
)

from utils import (
    assess_data_quality,
    format_verdict_for_display
)

from cache_utils import (
    get_cached_result,
    cache_result,
    clear_cache,
    get_cache_stats,
    is_redis_available
)

class AgentState(TypedDict):
    """Глобальное состояние агента для batch обработки."""
    
    # Входные данные
    user_queries: List[str]
    
    # Промежуточные данные (используем Annotated для полей, которые обновляются параллельно)
    tender_objects: Annotated[List[dict], operator.add]
    fact_sheets: Annotated[List[dict], operator.add]
    qdrant_matches: Annotated[List[List[dict]], operator.add]
    market_data: Annotated[List[List[dict]], operator.add]
    data_quality_scores: Annotated[List[dict], operator.add]
    
    # Финальные результаты
    final_verdicts: List[dict]
    
    # Флаги
    should_save_to_backup: bool
    needs_additional_search: bool
    
    # Ошибки
    errors: Annotated[List[str], operator.add]


# УЗЕЛ 1: BATCH SCRAPER
async def batch_scraper_node(state: AgentState) -> AgentState:
    """Скачивает документы для всех запросов (ASYNC)."""
    print("\n" + "="*60)
    print("📥 ШАГ 1: BATCH СКАЧИВАНИЕ ДОКУМЕНТОВ")
    print("="*60)
    
    queries = state["user_queries"]
    all_tender_objects = []
    
    for idx, query in enumerate(queries):
        print(f"\n🔍 Запрос {idx+1}/{len(queries)}: {query}")
        
        try:
            # ✅ ВАЖНО: ДОБАВИТЬ AWAIT
            tender_objs = await search_download_and_parse(query, target_count=1)
            
            if tender_objs:
                all_tender_objects.append(tender_objs[0])
                print(f"   ✅ Документ скачан")
            else:
                print(f"   ⚠️ Документ не найден")
                all_tender_objects.append({"error": "Не найдено", "query": query})
        
        except Exception as e:
            print(f"   ❌ Ошибка: {e}")
            all_tender_objects.append({"error": str(e), "query": query})
    
    return {"tender_objects": all_tender_objects}


# УЗЕЛ 2: BATCH EXTRACTOR
def batch_extractor_node(state: AgentState) -> AgentState:
    """Извлекает факты из всех документов."""
    print("\n" + "="*60)
    print("ШАГ 2: ИЗВЛЕЧЕНИЕ ФАКТОВ (Gemini)")
    print("="*60)
    
    tender_objects = state["tender_objects"]
    fact_sheets = []
    
    for idx, tender_obj in enumerate(tender_objects):
        print(f"\n Документ {idx+1}/{len(tender_objects)}")
        
        if "error" in tender_obj:
            print(f"   Пропуск (ошибка при скачивании)")
            fact_sheets.append({"error": tender_obj["error"]})
            continue
        
        try:
            fact_sheet = extract_facts_for_researcher(tender_obj)
            fact_sheets.append(fact_sheet)
            print(f"  Факты извлечены")
            print(fact_sheet)
        except Exception as e:
            print(f"   Ошибка извлечения: {e}")
            fact_sheets.append({"error": str(e)})
    
    return {"fact_sheets": fact_sheets}


# УЗЕЛ 3: BATCH QDRANT SEARCH
def batch_qdrant_search_node(state: AgentState) -> AgentState:
    """Ищет похожие риски для всех тендеров."""
    print("\n" + "="*60)
    print("🔎 ШАГ 3: ПОИСК ИСТОРИЧЕСКИХ РИСКОВ")
    print("="*60)
    
    fact_sheets = state["fact_sheets"]
    all_matches = []
    
    for idx, fact_sheet in enumerate(fact_sheets):
        print(f"\n🔍 Тендер {idx+1}/{len(fact_sheets)}")
        
        if "error" in fact_sheet:
            all_matches.append([])
            continue
        
        try:
            item_name = fact_sheet.get("meta_info", {}).get("predmet_zakupki", "")
            
            if not item_name:
                print(f"   ⚠️ Предмет закупки не указан")
                all_matches.append([])
                continue
            
            matches = check_similarity_risk(item_name)
            all_matches.append(matches or [])
            
            if matches:
                print(f"   ✅ Найдено {len(matches)} похожих случаев")
            else:
                print(f"   ✅ Похожих рисков нет")
        
        except Exception as e:
            print(f"   ❌ Ошибка: {e}")
            all_matches.append([])
    
    return {"qdrant_matches": all_matches}


# ============================================================================
# УЗЕЛ 4: BATCH MARKET SEARCH
# ============================================================================
def batch_market_search_node(state: AgentState) -> AgentState:
    """Ищет рыночные цены адаптивно."""
    print("\n" + "="*60)
    print("💰 ШАГ 4: АДАПТИВНЫЙ ПОИСК ЦЕН (+ Tavily)")
    print("="*60)
    
    fact_sheets = state["fact_sheets"]
    all_market_data = []
    
    for idx, fact_sheet in enumerate(fact_sheets):
        print(f"\n💵 Тендер {idx+1}/{len(fact_sheets)}")
        
        if "error" in fact_sheet:
            all_market_data.append([])
            continue
        
        try:
            market_data = search_prices_adaptive(fact_sheet)
            all_market_data.append(market_data)
            
            print(f"   ✅ Найдено {len(market_data)} ценовых предложений")
        
        except Exception as e:
            print(f"   ❌ Ошибка: {e}")
            all_market_data.append([])
    
    return {"market_data": all_market_data}


# УЗЕЛ 5: DATA QUALITY CHECK
def data_quality_check_node(state: AgentState) -> AgentState:
    """Оценивает качество собранных данных."""
    print("\n" + "="*60)
    print("📊 ШАГ 5: ОЦЕНКА КАЧЕСТВА ДАННЫХ")
    print("="*60)
    
    fact_sheets = state["fact_sheets"]
    market_data_list = state["market_data"]
    qdrant_matches_list = state["qdrant_matches"]
    
    quality_scores = []
    needs_extra_search = False
    
    for idx in range(len(fact_sheets)):
        fact_sheet = fact_sheets[idx]
        market_data = market_data_list[idx] if idx < len(market_data_list) else []
        qdrant_matches = qdrant_matches_list[idx] if idx < len(qdrant_matches_list) else []
        
        if "error" in fact_sheet:
            quality_scores.append({"score": 0, "reasons": ["Ошибка извлечения"]})
            continue
        
        quality = assess_data_quality(fact_sheet, market_data, qdrant_matches)
        quality_scores.append(quality)
        
        print(f"   Тендер {idx+1}: Качество {quality['score']}/100")
        
        if quality['score'] < 50:
            needs_extra_search = True
            print(f"      ⚠️ Низкое качество! Причины: {quality['reasons']}")
    
    return {
        "data_quality_scores": quality_scores,
        "needs_additional_search": needs_extra_search
    }


# УЗЕЛ 6: ADDITIONAL TAVILY SEARCH
def additional_search_node(state: AgentState) -> AgentState:
    """Выполняет дополнительный поиск через Tavily."""
    print("\n" + "="*60)
    print("🔎 ШАГ 6: ДОПОЛНИТЕЛЬНЫЙ ПОИСК (Tavily)")
    print("="*60)
    
    fact_sheets = state["fact_sheets"]
    market_data_list = list(state["market_data"])  # Создаем копию
    quality_scores = state["data_quality_scores"]
    
    for idx, quality in enumerate(quality_scores):
        if quality['score'] < 50:
            print(f"\n🔍 Дополнительный поиск для тендера {idx+1}...")
            
            fact_sheet = fact_sheets[idx]
            item_name = fact_sheet.get("meta_info", {}).get("predmet_zakupki", "")
            
            if item_name:
                extra_results = search_with_tavily(
                    f"{item_name} рыночная цена анализ Казахстан 2025",
                    max_results=5
                )
                
                market_data_list[idx].extend(extra_results)
                print(f"   ✅ Добавлено {len(extra_results)} источников")
    
    return {"market_data": market_data_list}


# УЗЕЛ 7: GPT-RESEARCHER
async def gpt_researcher_node(state: AgentState) -> AgentState:
    """GPT-Researcher синтезирует все данные (ASYNC)."""
    print("\n" + "="*60)
    print("🎯 ШАГ 7: GPT-RESEARCHER ФИНАЛЬНЫЙ АНАЛИЗ")
    print("="*60)
    
    try:
        # ✅ ВАЖНО: ДОБАВИТЬ AWAIT
        verdicts = await gpt_researcher_final_verdict(
            fact_sheets=state["fact_sheets"],
            qdrant_matches_list=state["qdrant_matches"],
            market_data_list=state["market_data"]
        )
        
        high_risk_count = sum(
            1 for v in verdicts 
            if v.get("overall_verdict") in ["HIGH_RISK", "CRITICAL_RISK"]
        )
        
        if high_risk_count > 0:
            print(f"\n⚠️ Обнаружено {high_risk_count} тендеров с высоким риском!")
            should_save = True
        else:
            should_save = False
        
        return {
            "final_verdicts": verdicts,
            "should_save_to_backup": should_save
        }
    
    except Exception as e:
        print(f"❌ Ошибка GPT-Researcher: {e}")
        import traceback
        traceback.print_exc()
        return {
            "errors": [str(e)],
            "final_verdicts": []
        }


# УЗЕЛ 8: SAVE TO BACKUP
def save_to_backup_node(state: AgentState) -> AgentState:
    """Сохраняет вердикты с высоким риском."""
    print("\n" + "="*60)
    print("💾 ШАГ 8: СОХРАНЕНИЕ В РЕЗЕРВНУЮ БАЗУ")
    print("="*60)
    
    verdicts = state["final_verdicts"]
    saved_count = 0
    
    for verdict in verdicts:
        if verdict.get("overall_verdict") in ["ВЫСОКИЙ_РИСК", "КРИТИЧЕСКИЙ_РИСК"]:
            try:
                save_approved_risk_to_qdrant(verdict)
                saved_count += 1
            except Exception as e:
                print(f"   ⚠️ Ошибка сохранения: {e}")
    
    print(f"\n✅ Сохранено вердиктов: {saved_count}/{len(verdicts)}")
    return {}


# CONDITIONAL EDGES
def should_do_additional_search(state: AgentState) -> Literal["search", "skip"]:
    """Определяет, нужен ли дополнительный поиск."""
    if state.get("needs_additional_search", False) and models.tavily_key:
        return "search"
    else:
        return "skip"


def should_save_to_backup(state: AgentState) -> Literal["save", "skip"]:
    """Определяет, нужно ли сохранять риски."""
    if state.get("should_save_to_backup", False):
        return "save"
    else:
        return "skip"


# УЛУЧШЕННОЕ ПОСТРОЕНИЕ ГРАФА
def build_graph():
    """Создаёт улучшенный LangGraph граф."""
    workflow = StateGraph(AgentState)
    
    # Добавляем узлы
    workflow.add_node("batch_scraper", batch_scraper_node)
    workflow.add_node("batch_extractor", batch_extractor_node)
    workflow.add_node("batch_qdrant_search", batch_qdrant_search_node)
    workflow.add_node("batch_market_search", batch_market_search_node)
    workflow.add_node("data_quality_check", data_quality_check_node)
    workflow.add_node("additional_search", additional_search_node)
    workflow.add_node("gpt_researcher", gpt_researcher_node)
    workflow.add_node("save_to_backup", save_to_backup_node)
    
    # Линейная часть
    workflow.add_edge(START, "batch_scraper")
    workflow.add_edge("batch_scraper", "batch_extractor")
    
    # Параллельные ветки
    workflow.add_edge("batch_extractor", "batch_qdrant_search")
    workflow.add_edge("batch_extractor", "batch_market_search")
    
    # Синхронизация
    workflow.add_edge("batch_qdrant_search", "data_quality_check")
    workflow.add_edge("batch_market_search", "data_quality_check")
    
    # Conditional Edge 1: Доп. поиск
    workflow.add_conditional_edges(
        "data_quality_check",
        should_do_additional_search,
        {
            "search": "additional_search",
            "skip": "gpt_researcher"
        }
    )
    
    workflow.add_edge("additional_search", "gpt_researcher")
    
    # Conditional Edge 2: Сохранение
    workflow.add_conditional_edges(
        "gpt_researcher",
        should_save_to_backup,
        {
            "save": "save_to_backup",
            "skip": END
        }
    )
    
    workflow.add_edge("save_to_backup", END)
    
    # Компилируем
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    
    return app


# ============================================================================
# ФУНКЦИЯ ЗАПУСКА
# ============================================================================
async def run_agent_batch(queries: List[str]):
    """Запускает агента для batch обработки с правильной структурой возврата."""
    print("\n" + "🚀"*30)
    print(f"ЗАПУСК BATCH АГЕНТА: {len(queries)} запросов")
    print("🚀"*30 + "\n")
    
    # ============================================================================
    # ШАГ 1: ПРОВЕРКА КЕША
    # ============================================================================
    if is_redis_available():
        print("✅ Redis cache: ACTIVE")
    else:
        print("⚠️ Redis cache: DISABLED")
    
    cached_verdicts = []
    queries_to_process = []
    query_cache_map = {}
    
    for query in queries:
        cached = get_cached_result(query, cache_type="batch_verdict")
        if cached:
            cached_verdicts.append(cached)
            query_cache_map[query] = cached
        else:
            queries_to_process.append(query)
    
    if cached_verdicts:
        print(f"🎯 Found {len(cached_verdicts)}/{len(queries)} results in cache")
    
    # Если всё в кеше - возвращаем сразу
    if not queries_to_process:
        print("✅ ALL RESULTS FROM CACHE!")
        return {
            "final_state": {  # ✅ ВАЖНО: Возвращаем в правильной структуре
                "final_verdicts": cached_verdicts,
                "from_cache": True,
                "cache_hit_rate": 100.0,
                "cache_stats": {
                    "total_queries": len(queries),
                    "cache_hits": len(cached_verdicts),
                    "cache_misses": 0,
                    "new_cached": 0
                }
            }
        }
    
    print(f"🔄 Processing {len(queries_to_process)} new queries...\n")
    
    # ============================================================================
    # ШАГ 2: ЗАПУСК ГРАФА
    # ============================================================================
    app = build_graph()
    
    initial_state = {
        "user_queries": queries_to_process,
        "tender_objects": [],
        "fact_sheets": [],
        "qdrant_matches": [],
        "market_data": [],
        "data_quality_scores": [],
        "final_verdicts": [],
        "should_save_to_backup": False,
        "needs_additional_search": False,
        "errors": []
    }
    
    config = {"configurable": {"thread_id": f"batch_{hash(tuple(queries_to_process))}"}}
    
    # ✅ КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Правильная обработка async generator
    final_state = None
    state_history = []
    
    try:
        async for state in app.astream(initial_state, config):
            print(f"📍 State update: {list(state.keys())}")
            state_history.append(state)
            final_state = state
    except Exception as e:
        print(f"❌ Ошибка графа: {e}")
        import traceback
        traceback.print_exc()
        return {
            "final_state": {
                "error": str(e),
                "final_verdicts": []
            }
        }
    
    # ============================================================================
    # ШАГ 3: ИЗВЛЕЧЕНИЕ РЕЗУЛЬТАТОВ
    # ============================================================================
    if not final_state:
        print("⚠️ No final state returned")
        return {
            "final_state": {
                "error": "No final state",
                "final_verdicts": []
            }
        }
    
    print(f"\n📦 Final state keys: {list(final_state.keys())}")
    
    # LangGraph возвращает {node_name: state_dict}
    state_values = list(final_state.values())
    
    if not state_values:
        print("⚠️ Empty state values")
        return {
            "final_state": {
                "error": "Empty state values",
                "final_verdicts": []
            }
        }
    
    last_state = state_values[-1]
    
    if last_state is None:
        print("⚠️ Last state is None! Searching history...")
        
        for state in reversed(state_history):
            state_vals = list(state.values())
            if state_vals and state_vals[-1] is not None:
                last_state = state_vals[-1]
                print(f"✅ Found valid state in history")
                break
        
        if last_state is None:
            return {
                "final_state": {
                    "error": "All states are None",
                    "final_verdicts": []
                }
            }
    
    new_verdicts = last_state.get("final_verdicts", []) if isinstance(last_state, dict) else []
    
    print(f"\n✅ Extracted {len(new_verdicts)} new verdicts")
    
    # ============================================================================
    # ШАГ 4: КЕШИРОВАНИЕ НОВЫХ РЕЗУЛЬТАТОВ
    # ============================================================================
    cache_success_count = 0
    for query, verdict in zip(queries_to_process, new_verdicts):
        if cache_result(query, verdict, cache_type="batch_verdict", ttl=86400):
            cache_success_count += 1
    
    if cache_success_count > 0:
        print(f"💾 Cached {cache_success_count}/{len(new_verdicts)} results")
    
    # ============================================================================
    # ШАГ 5: ОБЪЕДИНЕНИЕ РЕЗУЛЬТАТОВ
    # ============================================================================
    all_verdicts = []
    
    # Восстанавливаем исходный порядок
    for query in queries:
        if query in query_cache_map:
            verdict = query_cache_map[query]
            verdict["_from_cache"] = True
            all_verdicts.append(verdict)
        else:
            # Ищем в новых вердиктах
            found = False
            for verdict in new_verdicts:
                verdict_product = verdict.get("tender_summary", {}).get("product", "")
                if query.lower() in verdict_product.lower():
                    verdict["_from_cache"] = False
                    all_verdicts.append(verdict)
                    found = True
                    break
            
            if not found and new_verdicts:
                idx = queries_to_process.index(query) if query in queries_to_process else 0
                if idx < len(new_verdicts):
                    verdict = new_verdicts[idx]
                    verdict["_from_cache"] = False
                    all_verdicts.append(verdict)
    
    # ============================================================================
    # ШАГ 6: ФОРМИРОВАНИЕ ФИНАЛЬНОГО РЕЗУЛЬТАТА
    # ============================================================================
    cache_hit_rate = (len(cached_verdicts) / len(queries)) * 100 if queries else 0
    
    result = {
        "final_state": {  # ✅ ВАЖНО: Обернуть в final_state
            "final_verdicts": all_verdicts,
            "from_cache": len(cached_verdicts) > 0,
            "cache_hit_rate": round(cache_hit_rate, 1),
            "cache_stats": {
                "total_queries": len(queries),
                "cache_hits": len(cached_verdicts),
                "cache_misses": len(queries_to_process),
                "new_cached": cache_success_count
            }
        }
    }
    
    print(f"\n{'='*60}")
    print(f"✅ BATCH COMPLETED")
    print(f"   Total verdicts: {len(all_verdicts)}")
    print(f"   Cache hit rate: {cache_hit_rate:.1f}%")
    print(f"{'='*60}\n")
    
    return result


# ============================================================================
# ИНИЦИАЛИЗАЦИЯ QDRANT
# ============================================================================
def initialize_qdrant_database(csv_path: str = ""):
    """Загружает CSV в Qdrant."""
    print("\n" + "="*60)
    print("🗄️ ИНИЦИАЛИЗАЦИЯ БАЗЫ QDRANT")
    print("="*60)
    
    try:
        collections = models.qdrant_client.get_collections().collections
        collection_exists = any(c.name == models.collection_name for c in collections)
        
        if collection_exists:
            collection_info = models.qdrant_client.get_collection(models.collection_name)
            point_count = collection_info.points_count
            
            print(f"✅ Коллекция '{models.collection_name}' уже существует")
            print(f"📊 Количество записей: {point_count}")
            
            response = input("\n⚠️ Перезагрузить базу? (yes/no): ").lower()
            if response != "yes":
                print("⏭️ Пропускаю загрузку")
                return
    
    except Exception as e:
        print(f"⚠️ Не удалось проверить коллекцию: {e}")
    
    if not os.path.exists(csv_path):
        print(f"❌ Файл не найден: {csv_path}")
        return
    
    print(f"📂 Загружаю CSV: {csv_path}")
    
    try:
        df = pd.read_csv(csv_path, low_memory=False)
        print(f"✅ CSV загружен: {len(df)} строк")
        
        LoadDFinQdrant(df)
        print("✅ База Qdrant готова к работе!")
    
    except Exception as e:
        print(f"❌ Ошибка загрузки CSV: {e}")


# ГЛАВНАЯ ФУНКЦИЯ
def main():
    """Точка входа в программу с поддержкой управления кешем."""
    print("\n" + "="*60)
    print("💬 ВВОД ЗАПРОСОВ (BATCH MODE)")
    print("="*60)
    
    print("\nCache Options:")
    print("1. Use cache (default)")
    print("2. Clear cache and run fresh")
    print("3. Show cache statistics")
    
    cache_option = input("Select option (Enter for default): ").strip()
    
    if cache_option == "2":
        print("\n Clearing cache...")
        cleared = clear_cache("procurement:batch_verdict:*")
        print(f"Cleared {cleared} entries\n")
    
    elif cache_option == "3":
        print("\n📊 Cache Statistics:")
        stats = get_cache_stats()
        print(json.dumps(stats, indent=2, ensure_ascii=False))
        print()
        
        proceed = input("Continue with queries? (y/n): ").lower()
        if proceed != 'y':
            return None
    
    # ============================================================================
    # ВВОД ЗАПРОСОВ
    # ============================================================================
    print("Введите запросы через запятую (или Enter для примера):")
    user_input = input("🔍 Запросы: ").strip()
    
    if not user_input:
        queries = [
            "серверы Dell PowerEdge",
            "автобусы школьные Yutong"
        ]
        print(f"📝 Используются примеры: {queries}")
    else:
        queries = [q.strip() for q in user_input.split(",")]
    
    # ============================================================================
    # ЗАПУСК АГЕНТА
    # ============================================================================
    result = run_agent_batch(queries)
    
    # ============================================================================
    # ОБРАБОТКА РЕЗУЛЬТАТОВ
    # ============================================================================
    if not result:
        print("\n❌ Агент не вернул результат!")
        return None
    
    # Извлечение финального состояния
    try:
        final_state = None
        if isinstance(result, dict):
            state_values = list(result.values())
            if state_values:
                final_state = state_values[-1]
        
        if not final_state:
            print("\n❌ Не удалось извлечь финальное состояние")
            return None
        
        final_verdicts = final_state.get("final_verdicts", [])
        cache_stats = final_state.get("cache_stats", {})
        
        if final_verdicts:
            print("\n" + "="*60)
            print("📋 ФИНАЛЬНЫЕ ВЕРДИКТЫ:")
            print("="*60)
            
            # ✅ ПОКАЗЫВАЕМ СТАТИСТИКУ КЕША
            if cache_stats:
                print(f"\n📊 Cache Performance:")
                print(f"   Total queries: {cache_stats.get('total_queries', 0)}")
                print(f"   Cache hits: {cache_stats.get('cache_hits', 0)}")
                print(f"   Cache misses: {cache_stats.get('cache_misses', 0)}")
                print(f"   Hit rate: {final_state.get('cache_hit_rate', 0):.1f}%")
                print()
            
            for idx, verdict in enumerate(final_verdicts, 1):
                # Показываем из кеша или нет
                from_cache = verdict.get('_from_cache', False)
                cache_badge = "🎯 [CACHED]" if from_cache else "🆕 [NEW]"
                
                print(f"\n{cache_badge} {format_verdict_for_display(verdict)}")
        else:
            print("\n⚠️ Вердикты не найдены в результате")
        
        # ============================================================================
        # СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
        # ============================================================================
        output_file = "batch_agent_results.json"
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(
                    final_state, 
                    f, 
                    ensure_ascii=False, 
                    indent=2
                )
            print(f"\n💾 Результаты сохранены в {output_file}")
        except Exception as e:
            print(f"⚠️ Не удалось сохранить: {e}")
    
    except Exception as e:
        print(f"\n❌ Ошибка при обработке результата: {e}")
        import traceback
        traceback.print_exc()
    
    return result


# ============================================================================
# ЗАПУСК
# ============================================================================
if __name__ == "__main__":
    print("🚀"*30)
    print("СИСТЕМА АУДИТА ГОСЗАКУПОК (BATCH MODE)")
    print("🚀"*30 + "\n")
    
    # 1. Инициализируем модели
    print("🔧 Инициализация моделей...")
    _ = models
    if os.name == 'nt':
        # Принудительно устанавливаем WindowsSelectorEventLoopPolicy
        # Это исправляет NotImplementedError при работе с Playwright и Streamlit
        try:
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        except AttributeError:
            # Fallback для старых версий
            pass
    # 2. Загружаем базу рисков
    initialize_qdrant_database("D://Jigi//goszakup_contracts_FINAL_ANALYSIS.csv")
    
    # 3. Запускаем агента
    asyncio.run(main())