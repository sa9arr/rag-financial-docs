from google import genai
from google.genai.types import GenerateContentConfig, HttpOptions
from dotenv import load_dotenv
import os

with open("/home/ozymas/Pictures/rag-financial-docs/aapl_financial_data/2022 Q3 AAPL.md", "r") as f:
    context = f.read()

load_dotenv()
gemini_api_key = os.getenv("GENAI_API_KEY")

client = genai.Client(api_key=gemini_api_key)

def no_chunk_answer_query(query: str):
    
    

    response = client.models.generate_content(
        model = "gemini-2.5-flash",
        contents = query,
        config = GenerateContentConfig(
            system_instruction= [

                f"""You are an expert financial analyst. Your task is to provide accurate and factual answers to user questions based *only* on the financial report data provided in the context.
                Here is the context: {context}
                
                """
                                ]
        )
    )
    return response.text
    print(response.text)