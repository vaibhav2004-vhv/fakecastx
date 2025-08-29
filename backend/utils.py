# backend/utils.py
from langdetect import detect
def detect_lang(text: str) -> str:
    try:
        return detect(text)
    except:
        return "en"

def translate_to_en(text: str, src_lang: str) -> str:
    # stub: integrate Marian or external API later
    if src_lang == "en":
        return text
    return text

def fact_check_stub(text: str):
    # stubbed; replace with Google Fact Check / NewsAPI integration
    return {"checked": False, "sources": []}