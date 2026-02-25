"""
网络代理管理模块
用于在调用不同服务前动态切换代理状态
"""

import os
import subprocess
import contextlib
from typing import Optional


class ProxyManager:
    """代理管理器，用于在需要时启动或停止代理"""
    
    @staticmethod
    @contextlib.contextmanager
    def temporary_disable_proxy():
        """
        临时禁用代理的上下文管理器
        在此上下文中，所有代理环境变量都会被临时移除
        """
        # 保存原始的代理环境变量
        original_proxies = {}
        proxy_vars = ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'ALL_PROXY', 'all_proxy']
        
        for var in proxy_vars:
            if var in os.environ:
                original_proxies[var] = os.environ[var]
                del os.environ[var]
        
        try:
            yield
        finally:
            # 恢复原始的代理环境变量
            for var, value in original_proxies.items():
                os.environ[var] = value
    
    @staticmethod
    def run_system_command(command: str) -> Optional[str]:
        """
        执行系统命令
        """
        try:
            result = subprocess.run(
                command, 
                shell=True, 
                capture_output=True, 
                text=True, 
                timeout=30
            )
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                print(f"命令执行失败: {result.stderr}")
                return None
        except subprocess.TimeoutExpired:
            print(f"命令执行超时: {command}")
            return None
        except Exception as e:
            print(f"执行命令时出错: {e}")
            return None
    
    @staticmethod
    def stop_clash_proxy():
        """
        停止 clash 代理服务
        """
        # 尝试停止clash服务，命令可能因系统而异
        commands = [
            "sudo systemctl stop clash",  # systemd
            "brew services stop clash",   # macOS Homebrew
            "pkill -f clash",             # 通用方式
            "taskkill /f /im clash.exe"   # Windows
        ]
        
        for cmd in commands:
            if os.name == 'nt' and 'taskkill' not in cmd:
                continue  # 非Windows跳过Windows命令
            elif os.name != 'nt' and 'taskkill' in cmd:
                continue  # Windows跳过非Windows命令
                
            result = ProxyManager.run_system_command(cmd)
            if result is not None or "error" not in result.lower() if result else False:
                print(f"成功执行: {cmd}")
                return True
        
        print("无法停止clash代理服务，请手动操作")
        return False
    
    @staticmethod
    def start_clash_proxy():
        """
        启动 clash 代理服务
        """
        # 尝试启动clash服务，命令可能因系统而异
        commands = [
            "sudo systemctl start clash",  # systemd
            "brew services start clash",   # macOS Homebrew
            "nohup ~/clash &",             # 通用后台启动
            "start /b ~/clash.exe"         # Windows
        ]
        
        for cmd in commands:
            if os.name == 'nt' and 'start' not in cmd:
                continue  # 非Windows跳过Windows命令
            elif os.name != 'nt' and 'start' in cmd:
                continue  # Windows跳过非Windows命令
                
            result = ProxyManager.run_system_command(cmd)
            if result is not None:
                print(f"成功执行: {cmd}")
                return True
        
        print("无法启动clash代理服务，请手动操作")
        return False


@contextlib.contextmanager
def openrouter_api_call():
    """
    专门用于OpenRouter API调用的上下文管理器
    自动处理代理开关
    """
    print("准备调用OpenRouter API...")
    
    # 如果clash命令可用，尝试停止它
    if subprocess.run(['which', 'clash'], capture_output=True).returncode == 0:
        print("检测到clash命令，正在停止代理服务...")
        ProxyManager.stop_clash_proxy()
    else:
        # 否则只临时移除环境变量
        print("临时移除代理环境变量...")
    
    # 使用临时禁用代理的上下文
    with ProxyManager.temporary_disable_proxy():
        try:
            yield
        finally:
            # 如果clash命令可用，重新启动它
            if subprocess.run(['which', 'clash'], capture_output=True).returncode == 0:
                print("正在重新启动代理服务...")
                ProxyManager.start_clash_proxy()
            else:
                print("代理环境变量将在退出上下文后自动恢复")


# 使用示例
def example_usage():
    """
    使用示例
    """
    print("开始执行需要代理的操作（如下载模型）...")
    # 此时可以正常使用代理
    
    print("\n开始执行OpenRouter API调用...")
    with openrouter_api_call():
        # 在这里执行OpenRouter API调用
        print("执行OpenRouter API调用...")
        # 您的API调用代码放在这里
        pass
    
    print("\n继续执行需要代理的操作...")
    # 此时代理应该已经恢复