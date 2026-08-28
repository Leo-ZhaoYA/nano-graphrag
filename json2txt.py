import json
def extract_json_content(data):
    """递归提取 JSON 中的所有 content 字段并拼接"""
    text_output = ""
    # 提取当前层级的 content 字段
    if 'content' in data and data['content']:
        text_output += data['content'].strip() + "\n"  # 拼接并添加换行
    # 递归处理子数据
    if 'subdata' in data and data['subdata']:
        for subitem in data['subdata']:
            text_output += extract_json_content(subitem)  # 递归调用处理子数据
    return text_output
with open('pdf1.json', "r", encoding="utf-8") as f:
    json_data = json.load(f)
    file_content = extract_json_content(json_data)
    with open('pdf1.txt', 'w', encoding='utf-8') as f1:
        f1.write(file_content)