from flask import Flask, request, jsonify, render_template
from qa_system.database import Database
from qa_system.nlp_processor import NLPProcessor
import logging
from logging.handlers import RotatingFileHandler
import os

# 配置日志
def setup_logger():
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    file_handler = RotatingFileHandler(
        'logs/qa_system.log',
        maxBytes=1024 * 1024,  # 1MB
        backupCount=10
    )
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    
    logger = logging.getLogger('qa_system')
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    
    return logger

app = Flask(__name__)
logger = setup_logger()

# 确保在主线程中初始化数据库和NLP处理器
db = None
nlp = None

def init_app():
    global db, nlp
    try:
        db = Database('knowledge.db')
        nlp = NLPProcessor()
        logger.info("系统初始化成功")
    except Exception as e:
        logger.error(f"系统初始化失败: {str(e)}")
        raise

init_app()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/ask', methods=['POST'])
def ask_question():
    try:
        data = request.get_json()
        question = data.get('question')
        
        if not question:
            return jsonify({'error': '问题不能为空'}), 400
        
        logger.info(f"收到问题: {question}")
        
        # 获取所有问答对并查找最相似的答案
        qa_pairs = db.get_all_qa_pairs()
        logger.info(f"从数据库获取到 {len(qa_pairs)} 条问答对")
        
        if not qa_pairs:
            logger.warning("数据库中没有问答对数据")
            return jsonify({
                'answer': '抱歉，知识库还没有数据。',
                'similarity': 0
            })
        
        answer, similarity = nlp.find_most_similar(question, qa_pairs, threshold=0.5)  # 降低阈值试试
        
        logger.info(f"相似度: {similarity}, 答案: {answer}")
        
        if answer is None:
            logger.info(f"未找到答案: {question}")
            return jsonify({
                'answer': '抱歉，我没有找到相关的答案。',
                'similarity': 0
            })
        
        logger.info(f"找到案 (相似度: {similarity}): {answer}")
        return jsonify({
            'answer': answer,
            'similarity': similarity
        })
        
    except Exception as e:
        logger.error(f"处理问题时出错: {str(e)}")
        return jsonify({'error': '服务器内部错误'}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': '页面未找到'}), 404

@app.errorhandler(500)
def server_error(error):
    return jsonify({'error': '服务器内部错误'}), 500

if __name__ == '__main__':
    app.run(debug=True) 