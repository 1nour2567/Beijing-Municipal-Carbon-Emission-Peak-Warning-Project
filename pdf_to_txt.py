import PyPDF2
import os

def pdf_to_txt(pdf_path, txt_path=None):
    """
    将PDF文件转换为TXT文件
    :param pdf_path: PDF文件路径
    :param txt_path: 输出TXT文件路径，默认为同目录下同名TXT文件
    :return: 转换后的TXT文件路径
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF文件不存在: {pdf_path}")
    
    # 如果未指定输出路径，使用同目录下同名TXT文件
    if txt_path is None:
        txt_path = os.path.splitext(pdf_path)[0] + '.txt'
    
    # 打开PDF文件
    with open(pdf_path, 'rb') as pdf_file:
        # 创建PDF阅读器对象
        reader = PyPDF2.PdfReader(pdf_file)
        
        # 提取文本
        text = ''
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()
        
        # 写入TXT文件
        with open(txt_path, 'w', encoding='utf-8') as txt_file:
            txt_file.write(text)
    
    return txt_path

if __name__ == "__main__":
    # 示例用法
    pdf_file = "example.pdf"
    try:
        txt_file = pdf_to_txt(pdf_file)
        print(f"PDF转换成功！输出文件: {txt_file}")
    except Exception as e:
        print(f"转换失败: {str(e)}")
