import openai
import os
 
# 设置OpenAI API访问密钥
openai.api_key = "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"
 
# 调用ChatGPT API生成对话文本
response = openai.Completion.create(
    engine="davinci",
    prompt="Hello, how are you today?",
    max_tokens=50,
    n=1,
    stop=None,
    temperature=0.5,
)
 
# 从响应中提取生成的对话文本
text = response.choices[0].text.strip()
 
# 打印生成的对话文本
print(text)
