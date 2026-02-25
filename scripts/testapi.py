import os
from openai import OpenAI, APIConnectionError, APITimeoutError, APIError
import base64
import time
import httpx
from config.network_config import NetworkConfig

# 清除代理环境变量以避免 httpx 读取无效的 socks5 代理
# for key in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'ALL_PROXY', 'all_proxy', 'NO_PROXY', 'no_proxy']:
#     os.environ.pop(key, None)

def encode_image(image_path):
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

api_key = os.getenv('OPENROUTER_API_KEY', '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')
base_url = os.getenv('OPENROUTER_BASE_URL', "https://openrouter.ai/api/v1")
site_url = os.getenv('OPENROUTER_SITE_URL')
site_name = os.getenv('OPENROUTER_SITE_NAME')

img = encode_image('/root/autodl-tmp/ARLP/dataset/test/vlm_query_imgs/apple_agveuv_original.png')

# 使用 NetworkConfig 来创建绕过代理的 OpenRouter 客户端
client = NetworkConfig.get_openrouter_client(api_key=api_key, base_url=base_url)

# 尝试可用的视觉模型
# model_name = 'qwen/qwen-2.5-vl-72b-instruct'  # Qwen VL 模型通常可用
model_name = 'opengvlab/internvl3-78b'

for attempt in range(3):
    try:
        print(f"Attempt {attempt + 1}/3...")
        completion = client.chat.completions.create(
            model=model_name,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this image?"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}}
                ]
            }],
            extra_headers={
                "HTTP-Referer": site_url or "http://localhost",
                "X-Title": site_name or "Test",
            }
        )
        print(completion.choices[0].message.content)
        break
    except APIConnectionError as e:
        print(f"Connection error: {e}")
        if attempt < 2:
            time.sleep(2 ** attempt)  # 指数退避
        else:
            print("Please check your network/proxy settings. Try:\n  export HTTPS_PROXY=http://your-proxy:port")
    except APITimeoutError as e:
        print(f"Timeout error: {e}")
        if attempt < 2:
            time.sleep(2 ** attempt)
    except APIError as e:
        print(f"API error: {e}")
        break
    