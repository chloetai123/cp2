#apitest.py - to test if the GPT-4 API is correctly configured

from openai import OpenAI
import os

client = OpenAI()   # will read OPENAI_API_KEY from env

resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Say hello in one short sentence."}],
)

print(resp.choices[0].message.content)