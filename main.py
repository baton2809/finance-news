import pandas as pd
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from typing import List, Tuple
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Подключаем все переменные из окружения
load_dotenv()
LLM_API_KEY = os.getenv("LLM_API_KEY")
EMBEDDER_API_KEY = os.getenv("EMBEDDER_API_KEY")

# Глобальные переменные для кэша
knowledge_base = None
embeddings_cache = None


def get_embedding(text: str) -> List[float]:
    """
    Получение эмбеддинга для текста.
    
    Args:
        text: Текст для векторизации
        
    Returns:
        Список чисел - векторное представление текста
    """
    client = OpenAI(
        base_url="https://ai-for-finance-hack.up.railway.app/",
        api_key=EMBEDDER_API_KEY,
    )
    
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    
    return response.data[0].embedding


def load_knowledge_base(file_path: str = './train_data.csv') -> pd.DataFrame:
    """
    Загрузка базы знаний из CSV файла.
    
    Args:
        file_path: Путь к файлу с тренировочными данными
        
    Returns:
        DataFrame с базой знаний
    """
    logger.info(f"Загрузка базы знаний из {file_path}")
    
    # Загружаем данные с правильными разделителями
    df = pd.read_csv(file_path, sep=',', encoding='utf-8')
    
    logger.info(f"Загружено {len(df)} документов")
    return df


def create_embeddings_cache(knowledge_base: pd.DataFrame, cache_file: str = 'embeddings_cache.pkl') -> np.ndarray:
    """
    Создание кэша эмбеддингов для всех документов в базе знаний.
    
    Args:
        knowledge_base: DataFrame с базой знаний
        cache_file: Путь к файлу для сохранения кэша
        
    Returns:
        Numpy array с эмбеддингами
    """
    # Проверяем наличие кэша
    if os.path.exists(cache_file):
        logger.info(f"Загрузка эмбеддингов из кэша {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    logger.info("Создание эмбеддингов для базы знаний...")
    embeddings = []
    
    # Создаем составной текст для каждого документа
    for idx, row in tqdm(knowledge_base.iterrows(), total=len(knowledge_base), desc="Создание эмбеддингов"):
        # Комбинируем аннотацию и текст для лучшего поиска
        combined_text = f"{row['annotation']} {row['text'][:1000]}"  # Ограничиваем длину текста
        embedding = get_embedding(combined_text)
        embeddings.append(embedding)
    
    embeddings_array = np.array(embeddings)
    
    # Сохраняем кэш
    logger.info(f"Сохранение эмбеддингов в кэш {cache_file}")
    with open(cache_file, 'wb') as f:
        pickle.dump(embeddings_array, f)
    
    return embeddings_array


def retrieve_relevant_documents(question: str, knowledge_base: pd.DataFrame, 
                                embeddings: np.ndarray, top_k: int = 5) -> List[Tuple[int, float]]:
    """
    Поиск наиболее релевантных документов для вопроса.
    
    Args:
        question: Вопрос пользователя
        knowledge_base: DataFrame с базой знаний
        embeddings: Массив эмбеддингов документов
        top_k: Количество документов для возврата
        
    Returns:
        Список кортежей (индекс документа, score схожести)
    """
    # Получаем эмбеддинг вопроса
    question_embedding = np.array(get_embedding(question)).reshape(1, -1)
    
    # Вычисляем косинусное сходство
    similarities = cosine_similarity(question_embedding, embeddings)[0]
    
    # Получаем топ-K наиболее похожих документов
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    results = [(int(idx), float(similarities[idx])) for idx in top_indices]
    
    return results


def create_context_from_documents(indices: List[Tuple[int, float]], 
                                  knowledge_base: pd.DataFrame, 
                                  max_tokens: int = 3000) -> str:
    """
    Создание контекста из релевантных документов.
    
    Args:
        indices: Список кортежей (индекс документа, score)
        knowledge_base: DataFrame с базой знаний
        max_tokens: Максимальная длина контекста в токенах (примерно)
        
    Returns:
        Строка с контекстом
    """
    context_parts = []
    current_length = 0
    
    for idx, score in indices:
        doc = knowledge_base.iloc[idx]
        
        # Форматируем документ
        doc_text = f"Документ (релевантность: {score:.3f}):\n"
        doc_text += f"Аннотация: {doc['annotation']}\n"
        doc_text += f"Содержание: {doc['text']}\n"
        doc_text += "-" * 80 + "\n\n"
        
        # Примерная оценка длины (4 символа ~ 1 токен)
        estimated_tokens = len(doc_text) // 4
        
        if current_length + estimated_tokens > max_tokens:
            break
            
        context_parts.append(doc_text)
        current_length += estimated_tokens
    
    return "\n".join(context_parts)


def answer_generation(question: str, context: str = "") -> str:
    """
    Генерация ответа на вопрос с использованием контекста.
    
    Args:
        question: Вопрос пользователя
        context: Контекст из релевантных документов
        
    Returns:
        Ответ на вопрос
    """
    client = OpenAI(
        base_url="https://ai-for-finance-hack.up.railway.app/",
        api_key=LLM_API_KEY,
    )
    
    # Формируем промпт с контекстом
    if context:
        prompt = f"""Ты - эксперт по финансовой грамотности. Используй предоставленный контекст для ответа на вопрос пользователя.

КОНТЕКСТ:
{context}

ВОПРОС: {question}

ИНСТРУКЦИИ:
1. Отвечай точно и конкретно на поставленный вопрос
2. Используй информацию из контекста, если она релевантна
3. Если в контексте есть конкретные цифры, сроки или условия - обязательно укажи их
4. Ответ должен быть структурированным и понятным
5. Не выдумывай информацию, которой нет в контексте
6. Если в контексте недостаточно информации для полного ответа, используй свои знания, но укажи это

ОТВЕТ:"""
    else:
        prompt = f"Ответь на вопрос: {question}"
    
    response = client.chat.completions.create(
        model="openrouter/meta-llama/llama-3-70b-instruct",  # Используем более мощную модель
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ],
        temperature=0.3,  # Снижаем температуру для более точных ответов
        max_tokens=1000
    )
    
    return response.choices[0].message.content


def answer_with_rag(question: str, knowledge_base: pd.DataFrame, 
                    embeddings: np.ndarray, top_k: int = 3) -> str:
    """
    Генерация ответа с использованием RAG (Retrieval-Augmented Generation).
    
    Args:
        question: Вопрос пользователя
        knowledge_base: База знаний
        embeddings: Эмбеддинги документов
        top_k: Количество документов для контекста
        
    Returns:
        Ответ на вопрос
    """
    # 1. Поиск релевантных документов
    relevant_docs = retrieve_relevant_documents(question, knowledge_base, embeddings, top_k)
    
    # 2. Создание контекста
    context = create_context_from_documents(relevant_docs, knowledge_base)
    
    # 3. Генерация ответа с контекстом
    answer = answer_generation(question, context)
    
    return answer


if __name__ == "__main__":
    # Загружаем базу знаний
    logger.info("Инициализация системы...")
    knowledge_base = load_knowledge_base('./train_data.csv')
    
    # Создаем или загружаем эмбеддинги
    embeddings_cache = create_embeddings_cache(knowledge_base)
    
    # Считываем список вопросов
    logger.info("Загрузка вопросов...")
    questions = pd.read_csv('./questions.csv', sep='\t')
    questions_list = questions['Вопрос'].tolist()
    
    # Создаем список для хранения ответов
    answer_list = []
    
    # Проходимся по списку вопросов
    logger.info("Генерация ответов с использованием RAG...")
    for current_question in tqdm(questions_list, desc="Генерация ответов"):
        # Генерируем ответ с использованием RAG
        answer = answer_with_rag(
            current_question, 
            knowledge_base, 
            embeddings_cache,
            top_k=3  # Используем топ-3 документа для контекста
        )
        answer_list.append(answer)
    
    # Добавляем в данные список ответов
    questions['Ответы на вопрос'] = answer_list
    
    # Сохраняем submission
    logger.info("Сохранение результатов...")
    questions.to_csv('submission.csv', index=False, sep='\t')
    logger.info("Готово! Файл submission.csv создан.")
