from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Tuple
import logging
import os
import requests
from pathlib import Path

logger = logging.getLogger(__name__)

class NLPProcessor:
    def __init__(self, model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2'):
        """
        初始化NLP处理器
        
        Args:
            model_name: 使用的预训练模型名称，默认使用支持多语言的模型
        """
        try:
            # 设置模型缓存目录
            cache_dir = Path('models')
            cache_dir.mkdir(exist_ok=True)
            
            # 设置环境变量
            os.environ['TRANSFORMERS_CACHE'] = str(cache_dir)
            os.environ['HF_HOME'] = str(cache_dir)
            
            # 尝试加载模型
            logger.info(f"正在加载模型: {model_name}")
            self.model = SentenceTransformer(model_name, cache_folder=str(cache_dir))
            logger.info("模型加载成功")
            
        except Exception as e:
            logger.error(f"模型加载失败: {str(e)}")
            logger.info("使用备选的简单文本相似度计算方法")
            self.model = None
    
    def get_embedding(self, text: str) -> np.ndarray:
        """获取文本的嵌入向量"""
        if self.model is not None:
            try:
                return self.model.encode(text)
            except Exception as e:
                logger.error(f"获取嵌入向量失败: {str(e)}")
                return self._get_simple_embedding(text)
        return self._get_simple_embedding(text)
    
    def _get_simple_embedding(self, text: str) -> np.ndarray:
        """简单的文本向量化方法"""
        # 将文本转换为词袋表示
        words = text.lower().split()
        # 创建一个简单的向量（这里使用100维）
        vector = np.zeros(100)
        for i, word in enumerate(words):
            # 使用词的位置来设置向量的值
            vector[hash(word) % 100] += 1
        # 归一化向量
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector
    
    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """计算两个嵌入向量之间的余弦相似度"""
        try:
            return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
        except Exception as e:
            logger.error(f"计算相似度失败: {str(e)}")
            return 0.0
    
    def find_most_similar(self, 
                         query: str, 
                         candidates: List[Tuple], 
                         threshold: float = 0.5
                         ) -> Tuple[str, float]:
        """
        找到与查询最相似的问题及其答案
        
        Args:
            query: 用户的问题
            candidates: 候选QA对列表，每一项为(id, question, answer, embedding, created_at)
            threshold: 相似度阈值
            
        Returns:
            最相似的答案和相似度分数的元组，如果没有找到相似的问题则返回(None, 0)
        """
        query_embedding = self.get_embedding(query)
        logger.info(f"查询问题: {query}")
        logger.info(f"候选答案数量: {len(candidates)}")
        
        max_similarity = 0
        best_answer = None
        
        for row in candidates:
            try:
                # 解包数据库返回的元组
                id_, question, answer, embedding_str, created_at = row
                
                # 移除方括号并分割成数字列表
                embedding_str = embedding_str.strip('[]')
                embedding = np.array([float(x) for x in embedding_str.split(',')])
                
                similarity = self.calculate_similarity(query_embedding, embedding)
                logger.info(f"问题: {question}, 相似度: {similarity}")
                
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_answer = answer
                    logger.info(f"找到更好的匹配: {question} ({similarity})")
            except Exception as e:
                logger.error(f"处理嵌入向量时出错: {str(e)}")
                continue
        
        if max_similarity < threshold:
            logger.info(f"未找到足够相似的答案 (最大相似度: {max_similarity})")
            return None, 0
        
        logger.info(f"最终选择的答案 (相似度: {max_similarity}): {best_answer}")
        return best_answer, max_similarity 