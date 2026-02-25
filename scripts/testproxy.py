import os
import socket
import urllib.request
import subprocess
import sys

def check_port(host, port, timeout=2):
    """检查代理端口是否通畅（验证 Clash 内核是否在运行）"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception:
        return False

def test_network_request(url, timeout=5):
    """测试网络连通性"""
    try:
        req = urllib.request.Request(url)
        # urllib 默认会读取 os.environ 中的 http_proxy
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return True, response.status
    except Exception as e:
        return False, str(e)

def verify_scheme_a():
    print("="*60)
    print("🔍 开始验证方案 A (os.environ 操作代理)")
    print("="*60)

    # 1. 检查 Clash 内核端口 (默认 7890)
    print("\n1️⃣ 检查代理服务端 (Clash Core) 状态...")
    proxy_host = "127.0.0.1"
    proxy_port = 7890
    if check_port(proxy_host, proxy_port):
        print(f"   ✅ 端口 {proxy_host}:{proxy_port} 通畅 (Clash 内核正在运行)")
    else:
        print(f"   ❌ 端口 {proxy_host}:{proxy_port} 不通 (Clash 内核可能未启动！)")
        print("   ⚠️  注意：方案 A 只设置环境变量，不会启动 Clash 内核。")
        print("   ⚠️  如果内核没启动，设了环境变量也无法上网。")

    # 2. 测试 "关闭代理" 函数
    print("\n2️⃣ 测试 proxy_off (关闭代理)...")
    def proxy_off():
        for var in ['http_proxy', 'https_proxy', 'all_proxy', 
                    'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY']:
            os.environ.pop(var, None)
    
    proxy_off()
    current_proxy = os.environ.get('http_proxy', 'None')
    print(f"   当前 http_proxy 环境变量：{current_proxy}")
    if current_proxy == 'None':
        print("   ✅ 环境变量已成功清除")
    else:
        print("   ❌ 环境变量清除失败")

    # 3. 测试 "开启代理" 函数
    print("\n3️⃣ 测试 proxy_on (开启代理)...")
    def proxy_on(url="http://127.0.0.1:7890"):
        os.environ['http_proxy'] = url
        os.environ['https_proxy'] = url
        os.environ['all_proxy'] = url
    
    proxy_on()
    current_proxy = os.environ.get('http_proxy')
    print(f"   当前 http_proxy 环境变量：{current_proxy}")
    if current_proxy == "http://127.0.0.1:7890":
        print("   ✅ 环境变量已成功设置")
    else:
        print("   ❌ 环境变量设置失败")

    # 4. 测试网络请求 (关键步骤)
    print("\n4️⃣ 测试网络请求 (urllib)...")
    # 测试一个国内网站 (不需要代理)
    status, msg = test_network_request("http://www.baidu.com")
    print(f"   百度 (直连): {'✅ 通' if status else '❌ 不通'} ({msg})")
    
    # 测试一个国外网站 (需要代理，如果 Clash 配置正确)
    # 注意：如果 Clash 内核没启动，这里会失败
    status, msg = test_network_request("https://www.google.com")
    print(f"   Google (代理): {'✅ 通' if status else '❌ 不通'} ({msg})")
    if not status and "127.0.0.1:7890" in msg:
        print("   ⚠️  请求被代理拦截，说明环境变量生效了，但 Clash 内核可能挂了。")

    # 5. 测试子进程继承 (subprocess)
    print("\n5️⃣ 测试子进程继承 (subprocess)...")
    # 在子 shell 中打印环境变量，看是否继承了 Python 的设置
    result = subprocess.run(
        ["bash", "-c", "echo $http_proxy"],
        capture_output=True, text=True, env=os.environ.copy()
    )
    child_proxy = result.stdout.strip()
    print(f"   子进程 http_proxy: {child_proxy}")
    if child_proxy == "http://127.0.0.1:7890":
        print("   ✅ 子进程成功继承环境变量 (pip/git 等命令将生效)")
    else:
        print("   ❌ 子进程未继承 (需检查 env 传递)")

    print("\n" + "="*60)
    print("📊 验证结论:")
    if check_port("127.0.0.1", 7890) and os.environ.get('http_proxy'):
        print("✅ 方案 A 完全可行 (内核运行 + 变量设置成功)")
    elif not check_port("127.0.0.1", 7890):
        print("⚠️  方案 A 部分可行 (变量设置成功，但 Clash 内核未运行)")
        print("   建议：先手动运行一次终端命令 'clashon' 启动内核，之后再用 Python 管理。")
    else:
        print("❌ 方案 A 不可行 (环境变量设置失败)")
    print("="*60)

if __name__ == "__main__":
    verify_scheme_a()