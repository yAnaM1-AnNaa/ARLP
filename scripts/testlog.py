import textwrap
import logging
from pathlib import Path

def save_response(cot_response, logger):
    """
    保存响应到日志，正确格式化长句子
    
    Args:
        cot_response: 响应列表
        logger: 日志记录器
    """
    num_question = int(len(cot_response))
    q = 0
    
    while q < num_question:  # 修改条件，避免无限循环
        raw_sentence = str(cot_response[q])
        
        # 只有当句子长度超过指定宽度时才进行包装
        if len(raw_sentence) > 80:  # 使用更合理的宽度
            wrapped_sentence = textwrap.fill(
                raw_sentence,
                width=80,  # 使用合理的宽度
                break_long_words=True,  # 允许拆分长单词
                break_on_hyphens=True,  # 在连字符处断行
                subsequent_indent='    '  # 后续行缩进
            )
        else:
            wrapped_sentence = raw_sentence
        
        # 判断是问题还是回答
        if q % 2 == 0:
            logger.info(f'\nQuestion {q // 2 + 1}:')  # 修正问题编号
            logger.info(wrapped_sentence)
        else:
            logger.info(f'\nResponse {q // 2 + 1}:')  # 修正回答编号
            logger.info(wrapped_sentence)
        
        q += 1

def setup_logger_with_path(log_file_path):
    """设置日志记录器"""
    log_path = Path(log_file_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger("response_logger")
    logger.setLevel(logging.INFO)
    
    # 清除已有处理器
    logger.handlers.clear()
    
    # 文件处理器
    file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # 控制台处理器（可选）
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    return logger

# 使用示例
if __name__ == "__main__":
    logger = setup_logger_with_path("results/model_responses.log")

    # 测试数据
    cot_response = [
        "这是一个非常长的问题，包含了很多详细的信息和复杂的背景描述，需要模型进行深入的分析和解答。",
        "这是一个同样很长的模型回答，包含了详细的分析过程、推理步骤和最终的结论，对用户非常有价值。",
        "第二个问题是关于机器学习模型优化的复杂技术问题。",
        "对应的答案提供了多种优化策略和技术实现方案。"
    ]

    # 保存响应
    save_response(cot_response, logger)

    # 测试超长单词的包装效果
    long_word_sentence = "这是一个包含超长词汇的句子supercalifragilisticexpialidocious非常难以阅读需要适当的换行处理。"

    wrapped = textwrap.fill(
        long_word_sentence,
        width=30,
        break_long_words=True,
        break_on_hyphens=True,
        subsequent_indent='  '
    )

    print("原文:")
    print(long_word_sentence)
    print("\n包装后:")
    print(wrapped)