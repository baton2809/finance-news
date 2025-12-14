#baseline.py
import pandas as pd
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv
import os

# Подключаем все переменные из окружения
load_dotenv()
# Подключаем ключ для LLM-модели (теперь Perplexity)
LLM_API_KEY = os.getenv("LLM_API_KEY")


def answer_generation(question):
    """Генерация ответа с использованием Perplexity Sonar"""
    # Подключаемся к Perplexity API
    client = OpenAI(
        base_url="https://api.perplexity.ai",
        api_key=LLM_API_KEY,
    )
    # Формируем запрос к клиенту
    response = client.chat.completions.create(
        model="sonar-small-chat",
        messages=[
            {
                "role": "user",
                "content": f"Ответь на вопрос: {question}"
            }
        ],
        temperature=0.2,
        max_tokens=800,
    )
    # Формируем ответ на запрос и возвращаем его в результате работы функции
    return response.choices[0].message.content


if __name__ == "__main__":    
    # Считываем список вопросов
    questions = pd.read_csv('./questions.csv')
    # Выделяем список вопросов
    questions_list = questions['Вопрос'].tolist()
    # Создаем список для хранения ответов
    answer_list = []
    # Проходимся по списку вопросов
    for current_question in tqdm(questions_list, desc="Генерация ответов"):
        # Отправляем запрос на генерацию ответа
        answer = answer_generation(question=current_question)
        # Добавляем ответ в список
        answer_list.append(answer)
    # Добавляем в данные список ответов
    questions['Ответы на вопрос'] = answer_list
    # Сохраняем submission
    questions.to_csv('submission.csv', index=False)
