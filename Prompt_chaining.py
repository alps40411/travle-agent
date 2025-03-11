import logging
import os
import json
from datetime import datetime
from typing import Optional
import random
from pydantic import BaseModel, Field
from dotenv import load_dotenv
load_dotenv()  # 讀取 .env 檔案
from openai import OpenAI
import requests
import pandas as pd
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
token = os.getenv("TELEGRM_API_KEY")
import logging
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes


logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)


df = pd.read_csv('./Travel.csv')

data = df.drop(["Zone", 'Orgclass',"Travellinginfo","Opentime", "Picture1", "Picdescribe1", "Picture2", "Picdescribe2", "Picture3", "Picdescribe3", "Map", "Class1", "Class2","Class3", "Level", "Website", "Parkinginfo", "Parkinginfo_Px", "Parkinginfo_Py", "Ticketinfo", "Remarks", "Changetime"],axis=1)
df.head()

class PlaceResponse(BaseModel):
    name: str = Field(description="景點名稱")

    info: str = Field(description="景點介紹")

    place: str = Field(description="地址資訊")

class PlaceResponse(BaseModel):
    response: str = Field(
        description="A natural language response to the user's question."
    )

class WeatherResponse(BaseModel):
    temperature: float = Field(
        description="景點經緯度現在的天氣情況"
    )

    response: str = Field(
        description="A natural language response to the user's question."
    )

def search_places_by_location(location: str) -> list[dict]:
    """在 CSV 中搜尋與指定地點相關的景點"""
    if location =='台北':
        location = '臺北'

    results = data.query(f"Add.str.contains('{location}', case=False, na=False)", engine="python")['Name'].head(5).to_list()
    return results

def search_places_by_places_name(places_name: str) -> list[dict]:
    """在 CSV 中搜尋與指定地點相關的景點"""
    
    results = data.query(f"Name.str.contains('{places_name}', case=False, na=False)", engine="python").head(1).to_dict(orient='records')
    return results

def get_weather(places_name: str):
    results = data.query(f"Name.str.contains('{places_name}', case=False, na=False)", engine="python")[["Px", "Py"]].head(1).to_dict(orient='records')
    response = requests.get(
        f"https://api.open-meteo.com/v1/forecast?latitude={results[0]['Px']-90}&longitude={results[0]['Py']}&current=temperature_2m,wind_speed_10m&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m"
    )
    weather = response.json()
    return weather["current"]



def call_function(name, args):
    if name == "search_places_by_location":
        return search_places_by_location(**args), PlaceResponse
    
    elif name == 'search_places_by_places_name':
        return search_places_by_places_name(**args), PlaceResponse
    
    elif name == "get_weather":
        return get_weather(**args), WeatherResponse
    

    


# 定義 Function Calling 工具
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_places_by_location",
            "description": "搜尋指定地區的景點資訊",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "使用者想查詢的地點，例如城市名稱或景點名稱"}
                },
                "required": ["location"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_places_by_places_name",
            "description": "搜尋景點的相關資訊",
            "parameters": {
                "type": "object",
                "properties": {
                    "places_name": {"type": "string", "description": "藉由景點名稱，查找其相關資訊"}
                },
                "required": ["places_name"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "取得景點天氣相關資訊",
            "parameters": {
                "type": "object",
                "properties": {
                    "places_name": {"type": "string", "description": "藉由景點名稱，查找其相關資訊"}
                },  
                "required": ["places_name"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }
    

]



# 產生 AI 回應
def process_user_query(query: str):
    messages = [
        {"role": "system", "content": "你是一個旅遊 AI，根據使用者的需求提供景點資訊，但只能回答所獲得的資訊，不可自行產生"},
        {"role": "user", "content": query},
    ]
    
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=tools,
        tool_choice="auto" 
    )

    completion.model_dump()

    for tool_call in completion.choices[0].message.tool_calls:
        name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)

        


        messages.append(completion.choices[0].message)

        result, format = call_function(name, args)

        messages.append(
                {"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps(result)}
        )

        completion2 = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=messages,
        tools=tools,
        # response_format=format
        )
    


        return completion2.choices[0].message.content
    
    return '未找到相關景點，請嘗試不同的地點名稱。'
    
    
    
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=update.effective_chat.id, text = "您好！請輸入您想查詢的地點，例如：\n'請推薦台北的景點'")


async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE):

    user_message = update.message.text
    chat_id = update.effective_chat.id
    response = process_user_query(user_message)
    await update.message.reply_text(f"{response}")


token = '7830117616:AAFB4Wud4nf5VMYPfzmqiLA-pdtME0z34N0'

if __name__ == '__main__':
    application = ApplicationBuilder().token(token).build()
    
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo))
    
    application.run_polling()
