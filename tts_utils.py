from num2words import num2words
import re

def tts_friendly(text: str) -> str:
    text = text.replace("\u20b1", "₱")
    # Replace Peso symbols like ₱224 or ₱28 with words
    text = re.sub(r"₱(\d+)", lambda m: f"{num2words(int(m.group(1)))} pesos", text)
    # Convert fractional kWh values like 8.5kWh
    text = re.sub(r"(\d+(\.\d+)?)\s?kWh", lambda m: f"{m.group(1)} kilowatt hour{'s' if float(m.group(1)) != 1 else ''}", text)
    # Fix common abbreviations
    text = text.replace("/kWh", "per kilowatt hour")
    text = text.replace("/hour", "per hour")
    return text
