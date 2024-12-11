from qa_system.database import Database
from qa_system.nlp_processor import NLPProcessor

def init_knowledge_base():
    db = Database('knowledge.db')
    nlp = NLPProcessor()
    
    # 预设一些问答对
    qa_pairs = [
        ("人工智能是什么？", 
         "人工智能是计算机科学的一个分支，致力于开发能够模拟人类智能的系统。它包括机器学习、深度学习、自然语言处理等领域。"),
        
        ("什么是机器学习？", 
         "机器学习是人工智能的一个子领域，它使计算机系统能够通过经验自动改进。机器学习算法使用统计方法从数据中学习模式，而无需明确编程。"),
        
        ("Python是什么编程语言？", 
         "Python是一种高级、解释型、通用型编程语言。它强调代码的可读性，使用缩进来分隔代码块。Python支持多种编程范式，包括面向对象、命令式和函数式编程。"),
        
        ("深度学习是什么？", 
         "深度学习是机器学习的一个分支，使用多层神经网络来学习数据的表示。它在图像识别���自然语言处理等领域取得了突破性进展。"),
        
        ("什么是自然语言处理？", 
         "自然语言处理(NLP)是人工智能的一个分支，致力于使计算机理解和处理人类语言。它包括文本分类、机器翻译、问答系统等应用。")
    ]
    
    # 将问答对添加到数据库
    for question, answer in qa_pairs:
        try:
            embedding = nlp.get_embedding(question)
            success = db.add_qa_pair(question, answer, str(embedding.tolist()))
            if success:
                print(f"成功添加问答对: {question}")
            else:
                print(f"添加失败: {question}")
        except Exception as e:
            print(f"处理问题时出错: {str(e)}")

if __name__ == "__main__":
    init_knowledge_base() 