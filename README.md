# 🛡️ ForteGuard Ecosystem 360°

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-red.svg)](https://streamlit.io/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agentic-orange.svg)](https://langchain-ai.github.io/langgraph/)
[![Qdrant](https://img.shields.io/badge/Qdrant-Vector_DB-green.svg)](https://qdrant.tech/)
[![Redis](https://img.shields.io/badge/Redis-Caching-red.svg)](https://redis.io/)
[![Docker](https://img.shields.io/badge/Docker-Container-blue.svg)](https://www.docker.com/)

**ForteGuard Ecosystem** — это гибридная AI-платформа для автоматизированного аудита рисков в госзакупках и выявления транзакционного мошенничества.

Решение объединяет **Agentic RAG** для анализа неструктурированных документов и **Graph ML** для поведенческой биометрии, сокращая время проверки тендера с 40 минут до 30 секунд.

---

## 🚀 Основные возможности

### 1. 📑 AI-Procure (Аудит Госзакупок)
Агентная система, выполняющая работу аудитора в реальном времени:
* **Гибридный парсинг:** Автоматический сбор лотов (Playwright) + чтение сканов/PDF (Gemini Vision).
* **Semantic Risk Search:** Поиск смысловых совпадений с базой из **37,000+** исторических рисковых контрактов (Qdrant Vector DB).
* **Real-time Price Audit:** Проверка рыночных цен через **Google Shopping (SerpApi)** и **Tavily Search**. Выявляет завышение цен с учетом DDP и комплектации.
* **Z-Score Analysis:** Статистическое подтверждение аномалий цены.

---

## 🏗️ Архитектура

Проект построен на микросервисной архитектуре с использованием современных LLM-фреймворков.

```mermaid
graph TD
    User --> Streamlit["Streamlit UI"]
    Streamlit --> Cache{"Redis Cache"}
    Cache -- Miss --> Agent["LangGraph Agent"]
    
    subgraph "Procurement Engine"
        Agent --> Scraper["Playwright Parser"]
        Agent --> Vision["Gemini Vision (OCR)"]
        Agent --> VectorDB[("Qdrant: 37k History")]
        Agent --> Search["SerpApi / Tavily"]
    end

    
    Agent --> Analyst["GPT-Researcher (Synthesis)"]
    Analyst --> Streamlit
