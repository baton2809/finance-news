# Финансовый RAG Ассистент

Система Retrieval-Augmented Generation для ответов на вопросы по финансовой грамотности с оценкой качества через RAGAS метрики.

## Возможности

- Три режима работы RAG (v1/v2/v3)
- RAGAS метрики (Faithfulness, Answer Relevancy, Context Precision, Context Recall)
- Векторная база данных FAISS
- Веб-интерфейс Streamlit
- Docker инфраструктура с Nginx
- CLI для batch обработки

## Архитектура

```
┌───────────────────────────────┐
│     DOCKER COMPOSE STACK      │
├───────────────────────────────┤
│  ┌────────┐    ┌──────────┐  │
│  │ NGINX  │───▶│  WEB APP │  │
│  │(port 80)│   │(Streamlit)│  │
│  └────────┘    └──────────┘  │
│                               │
│       ┌──────────┐            │
│       │ CLI MODE │            │
│       └──────────┘            │
└───────────────────────────────┘
```

## Быстрый старт

### Требования

- Docker и Docker Compose
- DeepSeek API ключ
- 8GB RAM

### Запуск

```bash
# 1. Клонировать репозиторий
git clone <repo-url>
cd finance-news

# 2. Создать .env файл
cp .env.example .env
nano .env  # Добавить LLM_API_KEY=sk-your-key

# 3. Запустить
docker-compose up --build

# 4. Открыть http://localhost
```

### CLI режим

```bash
# Обработать 10 вопросов
docker-compose --profile cli run cli python main.py --mode v2 --limit 10

# Все вопросы
docker-compose --profile cli run cli python main.py --mode v3
```

## Структура проекта

```
finance-news/
├── main.py                          # RAG pipeline
├── app.py                           # Streamlit UI
├── generate_references.py           # Генерация референсов
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
│
├── nginx/
│   └── nginx.conf
│
├── .env                            # API ключи
├── .env.example
│
├── train_data.csv                  # База знаний (5.8 MB)
├── questions.csv
├── questions_with_references.csv   # 100 вопросов с референсами
│
├── submission.csv                  # Результаты
├── metrics_detailed.csv
│
├── faiss_index_v1_e5small.bin
├── faiss_index_v2_e5small.bin
├── faiss_meta_v1_e5small.pkl
└── faiss_meta_v2_e5small.pkl
```

## Режимы RAG

### v1: Document-Level

- Векторизация целых документов
- Быстрый, но менее точный
- Context Precision: ~0.65

### v2: Chunk-Based (Рекомендуется)

- Разбиение на фрагменты (1000 символов, overlap 200)
- Баланс скорость/качество
- Context Precision: ~0.75

### v3: Chunk + LLM Reranking

- v2 + переранжирование через LLM
- Лучшее качество, медленнее
- Context Precision: ~0.85

## RAGAS Метрики

| Метрика | Требует референс | Диапазон | Цель |
|---------|-----------------|----------|------|
| Faithfulness | Нет | 0.0-1.0 | >0.85 |
| Answer Relevancy | Нет | 0.0-1.0 | >0.80 |
| Context Precision | Да | 0.0-1.0 | >0.75 |
| Context Recall | Да | 0.0-1.0 | >0.80 |

**Faithfulness**: Проверяет галлюцинации (ответ только из контекста)
**Answer Relevancy**: Релевантность ответа вопросу
**Context Precision**: Точность извлечения (мало шума)
**Context Recall**: Полнота извлечения (нет пропусков)

## Docker инфраструктура

### Сервисы

**NGINX (port 80)**
- Reverse proxy
- Rate limiting (10 req/sec)
- WebSocket support для Streamlit
- Health check: `/health`

**WEB (Streamlit, port 8501)**
- Веб-интерфейс RAG
- Кэширование модели/индекса
- Volumes: код, кэш, HuggingFace модели

**CLI (Batch Processing)**
- Запуск через `--profile cli`
- Массовая обработка вопросов

## CLI Использование

```bash
# Аргументы
python main.py [--mode v1|v2|v3] [--limit N]

# Примеры
python main.py --mode v1 --limit 30
python main.py --mode v2
python main.py --mode v3 --limit 10
```

Выходные файлы:
- `submission.csv` - вопросы + ответы
- `metrics_detailed.csv` - детальные метрики

## Технологический стек

| Компонент | Технология | Версия |
|-----------|------------|--------|
| Embeddings | multilingual-e5-small | 384 dim |
| Vector DB | FAISS IndexFlatIP | 1.7.0+ |
| LLM | DeepSeek API | deepseek-chat |
| Evaluation | RAGAS | 0.4.2+ |
| Web UI | Streamlit | 1.28.0+ |
| Infrastructure | Docker Compose + Nginx | - |

## Мониторинг

```bash
# Статус
docker-compose ps

# Логи
docker-compose logs web
docker-compose logs nginx
docker-compose logs -f web

# Health checks
curl http://localhost/health
```

## Troubleshooting

**RAGAS метрики 0.0 или NaN**
- Причина: DeepSeek не поддерживает `n>1`
- Решение: Проверьте `AnswerRelevancy(strictness=1)` в main.py

**Cannot load embedding model**
- Проверьте интернет
- Пересоберите: `docker-compose build --no-cache web`

**Streamlit не открывается**
- Проверьте: `docker-compose ps`
- Логи: `docker-compose logs web`
- Health: `curl http://localhost/health`

**API key not configured**
- Проверьте `.env`: `cat .env`
- Перезапустите: `docker-compose down && docker-compose up`

## Производительность

| Режим | Время (сек) | API calls |
|-------|-------------|-----------|
| v1 | 2-3 | 2 |
| v2 | 3-4 | 2 |
| v3 | 5-7 | 3 |

Оптимизация:
- Для скорости: используйте v2, уменьшите top_k
- Для качества: v3, top_k=20-30, final_k=5
- Для стоимости: избегайте v3, используйте batch CLI

---

Команда A3R | Январь 2026
