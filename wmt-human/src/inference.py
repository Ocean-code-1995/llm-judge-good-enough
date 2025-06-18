import openai
import os
import dotenv

dotenv.load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_response(prompt, model="gpt-4o", temperature=0.7, max_tokens=500):
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "system", "content": "You are a helpful assistant."},
                  {"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response["choices"][0]["message"]["content"]


if __name__ == "__main__":
    prompt = "What are the key challenges in evaluating Retrieval-Augmented Generation (RAG) systems?"
    response = generate_response(prompt)
    print(response)
