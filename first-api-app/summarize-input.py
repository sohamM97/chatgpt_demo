import os

import dotenv
import openai

dotenv.load_dotenv("../.env")

openai.api_key = os.getenv("OPENAI_API_KEY")

user_input = input("Enter text that should be summarized: ")

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {
            "role": "system",
            "content": "You are a text summarizer. Your goal is to summarize the text "
            "that is given to you by the user.",
        },
        {"role": "user", "content": user_input},
    ],
    temperature=0,
)

ai_response = response.choices[0].message.content
print("Summarized text:\n\n", ai_response)
