import sqlite3
from typing import List, Tuple
import logging
from threading import Lock

logger = logging.getLogger('qa_system')


class Database:
    _instance = None
    _lock = Lock()

    def __init__(self, db_path: str):
        """初始化数据库连接"""
        self.db_path = db_path
        self.create_tables()

    def get_connection(self):
        """获取线程安全的数据库连接"""
        if not hasattr(self, '_conn'):
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        return self._conn

    def get_cursor(self):
        """获取数据库游标"""
        return self.get_connection().cursor()

    def create_tables(self):
        """创建必要的数据表"""
        try:
            with self._lock:
                cursor = self.get_cursor()
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS qa_pairs (
                    id INTEGER PRIMARY KEY,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    embedding TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                ''')
                self.get_connection().commit()
                logger.info("数据表创建/确认成功")
        except Exception as e:
            logger.error(f"创建数据表失败: {str(e)}")
            raise

    def add_qa_pair(self, question: str, answer: str, embedding: str) -> bool:
        """添加新的问答对"""
        try:
            with self._lock:
                cursor = self.get_cursor()
                cursor.execute(
                    'INSERT INTO qa_pairs (question, answer, embedding) VALUES (?, ?, ?)',
                    (question, answer, str(embedding))
                )
                self.get_connection().commit()
                logger.info(f"成功添加新的问答对: {question}")
                return True
        except Exception as e:
            logger.error(f"添加问答对失败: {str(e)}")
            return False

    def get_all_qa_pairs(self) -> List[Tuple]:
        """获取所有问答对"""
        try:
            with self._lock:
                cursor = self.get_cursor()
                results = cursor.execute('SELECT * FROM qa_pairs').fetchall()
                logger.info(f"从数据库获取到 {len(results)} 条问答对")
                return results
        except Exception as e:
            logger.error(f"获取问答对失败: {str(e)}")
            return []

    def search_similar_questions(self, question: str, threshold: float = 0.8) -> List[Tuple]:
        """搜索相似的问题"""
        try:
            with self._lock:
                cursor = self.get_cursor()
                return cursor.execute(
                    'SELECT * FROM qa_pairs WHERE similarity(question, ?) > ?',
                    (question, threshold)
                ).fetchall()
        except Exception as e:
            logger.error(f"搜索相似问题失败: {str(e)}")
            return []

    def __del__(self):
        """确保关闭数据库连接"""
        try:
            if hasattr(self, '_conn'):
                with self._lock:
                    self._conn.close()
                    logger.info("数据库连接已关闭")
        except Exception as e:
            logger.error(f"关闭数据库连接失败: {str(e)}")
