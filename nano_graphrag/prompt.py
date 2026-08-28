# This file contains the prompts for the tasks in the GraphRAG dataset.
GRAPH_FIELD_SEP = "<SEP>"
PROMPTS = {}
PROMPTS["fail_response"] = "抱歉，我无法回答这个问题。"
PROMPTS["process_tickers"] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|>"
PROMPTS["DEFAULT_RECORD_DELIMITER"] = "##"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"

# DOMAIN_IN_SPECIFIC指定输入文本和可能提问的领域，注意列表长度最好不要超过5
# PROMPTS["DOMAIN_IN_SPECIFIC"] = ["雷达信号处理", "芯片", "电路板", "编程", "数字电子技术", "模拟电子技术"]
PROMPTS["DOMAIN_IN_SPECIFIC"] = ["国际民用航空", "航空电信", "通信、导航、监视与无线电"]
PROMPTS["DEFAULT_ENTITY_TYPES"] = [
    "组织",
    "人名",
    "地点",
    "事件",
    "技术",
    "产品",
    "概念",
    "算法",
    "系统",
    "专利",
    "论文",
    "设备",
    "数据标准",
]


PROMPTS["default_text_separator"] = [
    # Paragraph separators
    "\n\n",
    "\r\n\r\n",
    # Line breaks
    "\n",
    "\r\n",
    # Sentence ending punctuation
    "。",  # Chinese period
    "．",  # Full-width dot
    ".",  # English period
    "！",  # Chinese exclamation mark
    "!",  # English exclamation mark
    "？",  # Chinese question mark
    "?",  # English question mark
    # Whitespace characters
    " ",  # Space
    "\t",  # Tab
    "\u3000",  # Full-width space
    # Special characters
    "\u200b",  # Zero-width space (used in some Asian languages)
]


PROMPTS[
    "e_r_extraction"
] = """
# 实体与关系提取任务

您的任务是"给定一个可能与此活动相关的文本文件和一个实体类型列表，从文本中识别所有这些类型的实体以及这些实体之间的所有关系"。
严格按照以下格式要求操作。

## 输入格式
- entity_types: 实体类型列表
- text: 需要分析的文本内容

## 输出格式要求(必须严格遵循)

1. 每个实体必须使用确切的格式: 
   ("entity"{tuple_delimiter}实体名称{tuple_delimiter}实体类型{tuple_delimiter}实体描述){record_delimiter}
   
2. 每个关系必须使用确切的格式: 
   ("relationship"{tuple_delimiter}源实体名称{tuple_delimiter}目标实体名称{tuple_delimiter}关系描述{tuple_delimiter}关系强度数字){record_delimiter}

3. 所有条目之间必须使用 {record_delimiter} 作为分隔符，每个条目后面都跟随 {record_delimiter}

4. 完成后必须添加 {completion_delimiter} 标记

## 警告

- 不要使用任何其他格式或添加额外内容
- 不要使用 "实体:"、"关系:" 等标签
- 实体类型必须从提供的列表中选择
- 实体描述指: 实体属性和活动的详细描述
- 关系描述指: 解释为什么认为源实体和目标实体彼此相关
- 关系强度数字必须是1-10之间的整数，表示源实体和目标实体之间关系的强度，分数值越大表示强度越高


## 示例
### 示例1
#### 输入示例: 
entity_types: [数据预处理方法, 图像处理结果, 数据结构, 图像处理方法]
text: 骨架化与谱线分离方法:
1. 为了避免曲线“毛刺”对骨架提取的不利影响，可先进行高斯平滑处理，并设置合适阈值将其转化为二值图像，在此基础上再提取曲线骨架。通过提取 m-D 曲线的骨架，就可达到抑制距离像旁瓣引起的曲线变“粗”的现象。
2. 根据形态学图像处理理论，骨架提取可由“腐蚀”运算和“开”运算实现。抑制 m-D 曲线的距离像旁瓣引起的曲线变“粗”的现象，提取曲线骨架。

#### 正确输出示例: 
("entity"{tuple_delimiter}"高斯平滑"{tuple_delimiter}数据预处理方法"{tuple_delimiter}"用于减少噪声，提高骨架提取的准确性"){record_delimiter}
("entity"{tuple_delimiter}"二值图像"{tuple_delimiter}"图像处理结果"{tuple_delimiter}"将原始图像转换为二值形式以简化后续分析"){record_delimiter}
("entity"{tuple_delimiter}"曲线骨架"{tuple_delimiter}"数据结构"{tuple_delimiter}"表示曲线的主要形状或轮廓"){record_delimiter}
("entity"{tuple_delimiter}"形态学图像处理理论"{tuple_delimiter}"图像处理方法"{tuple_delimiter}"包括腐蚀和开运算，用于提取骨架"){record_delimiter}
("relationship"{tuple_delimiter}"高斯平滑"{tuple_delimiter}"二值图像"{tuple_delimiter}"为曲线骨架提取提供更清晰的输入"{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"二值图像"{tuple_delimiter}"形态学图像处理理论"{tuple_delimiter}"为后续的骨架提取提供明确边界"{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"曲线骨架"{tuple_delimiter}"形态学图像处理理论"{tuple_delimiter}"描述了原始曲线的主要特征"{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"形态学图像处理理论"{tuple_delimiter}"高斯平滑"{tuple_delimiter}"为曲线骨架提取提供理论基础"{tuple_delimiter}7){record_delimiter}
{completion_delimiter}

### 示例2
#### 输入示例: 
entity_types: [数据结构, 算法]
text: 快速排序是一种高效排序算法，采用分治策略。链表是存储数据的常用结构。

#### 正确输出示例: 
("entity"{tuple_delimiter}快速排序{tuple_delimiter}算法{tuple_delimiter}一种高效的排序算法，采用分治策略){record_delimiter}
("entity"{tuple_delimiter}链表{tuple_delimiter}数据结构{tuple_delimiter}存储数据的常用结构){record_delimiter}
("entity"{tuple_delimiter}分治{tuple_delimiter}策略{tuple_delimiter}一种在处理问题时的采用方法，将问题拆分为多个子问题逐个处理){record_delimiter}
("relationship"{tuple_delimiter}快速排序{tuple_delimiter}链表{tuple_delimiter}快速排序可以应用于链表结构上{tuple_delimiter}7){record_delimiter}
{completion_delimiter}

## 当前任务
entity_types: {entity_types}
text: {input_text}

## 格式检查提示
处理完成后，请检查: 
1. 所有实体格式是否为 ("entity"{tuple_delimiter}实体名称{tuple_delimiter}实体类型{tuple_delimiter}实体描述){record_delimiter}
2. 所有关系格式是否为 ("relationship"{tuple_delimiter}源实体名称{tuple_delimiter}目标实体名称{tuple_delimiter}关系描述{tuple_delimiter}关系强度数字){record_delimiter}
3. 实体类型是否从提供的entity_types列表中选择
4. 末尾是否添加了 {completion_delimiter}

"""

PROMPTS[
    "e_extraction"
] = """
# 实体提取任务

您的任务是"给定一个可能与此活动相关的文本文件和一个实体类型列表，从文本中识别所有这些类型的实体。如果文本中包含了对一些实体的定义，你需要着重提取识别这些实体。但也别忘了提取文本中的其他实体"。
严格按照以下格式要求操作。

## 输入格式
- entity_types: 实体类型列表
- text: 需要分析的文本内容

## 输出格式要求(必须严格遵循)

1. 每个实体必须使用确切的格式: 
   ("entity"{tuple_delimiter}实体名称{tuple_delimiter}实体类型{tuple_delimiter}实体描述){record_delimiter}

2. 所有条目之间必须使用 {record_delimiter} 作为分隔符，每个条目后面都跟随 {record_delimiter}

3. 完成后必须添加 {completion_delimiter} 标记

## 警告

- 不要使用任何其他格式或添加额外内容
- 不要使用 "实体:"等标签
- 实体类型必须从提供的列表中选择
- 实体描述指: 实体属性和活动的详细描述

## 示例
### 示例1
#### 输入示例: 
entity_types: [数据预处理方法, 图像处理结果, 数据结构, 图像处理方法]
text: 骨架化与谱线分离方法:
1. 为了避免曲线“毛刺”对骨架提取的不利影响，可先进行高斯平滑处理，并设置合适阈值将其转化为二值图像，在此基础上再提取曲线骨架。通过提取 m-D 曲线的骨架，就可达到抑制距离像旁瓣引起的曲线变“粗”的现象。
2. 根据形态学图像处理理论，骨架提取可由“腐蚀”运算和“开”运算实现。抑制 m-D 曲线的距离像旁瓣引起的曲线变“粗”的现象，提取曲线骨架。

#### 正确输出示例: 
("entity"{tuple_delimiter}"高斯平滑"{tuple_delimiter}数据预处理方法"{tuple_delimiter}"用于减少噪声，提高骨架提取的准确性"){record_delimiter}
("entity"{tuple_delimiter}"二值图像"{tuple_delimiter}"图像处理结果"{tuple_delimiter}"将原始图像转换为二值形式以简化后续分析"){record_delimiter}
("entity"{tuple_delimiter}"曲线骨架"{tuple_delimiter}"数据结构"{tuple_delimiter}"表示曲线的主要形状或轮廓"){record_delimiter}
("entity"{tuple_delimiter}"形态学图像处理理论"{tuple_delimiter}"图像处理方法"{tuple_delimiter}"包括腐蚀和开运算，用于提取骨架"){record_delimiter}
{completion_delimiter}


### 示例2
#### 输入示例: 
entity_types: [数据结构, 算法]
text: 快速排序是一种高效排序算法，采用分治策略。链表是存储数据的常用结构。

#### 正确输出示例: 
("entity"{tuple_delimiter}快速排序{tuple_delimiter}算法{tuple_delimiter}一种高效的排序算法，采用分治策略){record_delimiter}
("entity"{tuple_delimiter}链表{tuple_delimiter}数据结构{tuple_delimiter}存储数据的常用结构){record_delimiter}
("entity"{tuple_delimiter}分治{tuple_delimiter}策略{tuple_delimiter}一种在处理问题时的采用方法，将问题拆分为多个子问题逐个处理){record_delimiter}
{completion_delimiter}

## 当前任务
entity_types: {entity_types}
text: {input_text}

## 格式检查提示
处理完成后，请检查: 
1. 所有实体格式是否为 ("entity"{tuple_delimiter}实体名称{tuple_delimiter}实体类型{tuple_delimiter}实体描述){record_delimiter}
2. 实体类型是否从提供的entity_types列表中选择
3. 末尾是否添加了 {completion_delimiter}

"""


PROMPTS[
    "r_extraction"
] = """
# 关系提取任务

您的任务是"给定一个文本文件、一个实体类型列表和一个已经识别到的实体列表，从文本中找到已经识别到的实体列表中的这些实体，然后返回这些实体之间的所有关系。"。
严格按照以下格式要求操作。

## 输入格式
- entity_types: 实体类型列表
- entity_lists: 实体列表
- text: 需要分析的文本内容

注意: 实体列表entity_lists中各实体在输入时已经预先被格式化为 ("entity"{tuple_delimiter}实体名称{tuple_delimiter}实体类型{tuple_delimiter}实体描述){record_delimiter} 格式。

## 输出格式要求(必须严格遵循)

1. 每个关系必须使用确切的格式: 
   ("relationship"{tuple_delimiter}源实体名称{tuple_delimiter}目标实体名称{tuple_delimiter}关系描述{tuple_delimiter}关系强度数字){record_delimiter}

2. 所有条目之间必须使用 {record_delimiter} 作为分隔符，每个条目后面都跟随 {record_delimiter}

3. 完成后必须添加 {completion_delimiter} 标记

## 警告

- 不要使用任何其他格式或添加额外内容
- 不要使用 "实体:"、"关系:" 等标签
- 关系描述指: 解释为什么认为源实体和目标实体彼此相关
- 关系强度数字必须是1-10之间的整数，表示源实体和目标实体之间关系的强度，分数值越大表示强度越高


## 示例
### 示例1
#### 输入示例: 
entity_types: [数据预处理方法, 图像处理结果, 数据结构, 图像处理方法]
entity_lists:
("entity"{tuple_delimiter}"高斯平滑"{tuple_delimiter}数据预处理方法"{tuple_delimiter}"用于减少噪声，提高骨架提取的准确性"){record_delimiter}
("entity"{tuple_delimiter}"二值图像"{tuple_delimiter}"图像处理结果"{tuple_delimiter}"将原始图像转换为二值形式以简化后续分析"){record_delimiter}
("entity"{tuple_delimiter}"曲线骨架"{tuple_delimiter}"数据结构"{tuple_delimiter}"表示曲线的主要形状或轮廓"){record_delimiter}
("entity"{tuple_delimiter}"形态学图像处理理论"{tuple_delimiter}"图像处理方法"{tuple_delimiter}"包括腐蚀和开运算，用于提取骨架"){record_delimiter}
{completion_delimiter}
text: 骨架化与谱线分离方法:
1. 为了避免曲线“毛刺”对骨架提取的不利影响，可先进行高斯平滑处理，并设置合适阈值将其转化为二值图像，在此基础上再提取曲线骨架。通过提取 m-D 曲线的骨架，就可达到抑制距离像旁瓣引起的曲线变“粗”的现象。
2. 根据形态学图像处理理论，骨架提取可由“腐蚀”运算和“开”运算实现。抑制 m-D 曲线的距离像旁瓣引起的曲线变“粗”的现象，提取曲线骨架。

#### 正确输出示例: 
("relationship"{tuple_delimiter}"高斯平滑"{tuple_delimiter}"二值图像"{tuple_delimiter}"为曲线骨架提取提供更清晰的输入"{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"二值图像"{tuple_delimiter}"形态学图像处理理论"{tuple_delimiter}"为后续的骨架提取提供明确边界"{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"曲线骨架"{tuple_delimiter}"形态学图像处理理论"{tuple_delimiter}"描述了原始曲线的主要特征"{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"形态学图像处理理论"{tuple_delimiter}"高斯平滑"{tuple_delimiter}"为曲线骨架提取提供理论基础"{tuple_delimiter}7){record_delimiter}
{completion_delimiter}


## 当前任务
entity_types: {entity_types}
entity_lists: {entity_lists}
text: {input_text}

## 格式检查提示
处理完成后，请检查: 
1. 所有关系格式是否为 ("relationship"{tuple_delimiter}源实体名称{tuple_delimiter}目标实体名称{tuple_delimiter}关系描述{tuple_delimiter}关系强度数字){record_delimiter}
2. 末尾是否添加了 {completion_delimiter}

"""


PROMPTS[
    "summarize_entity_descriptions"
] = """
# 实体描述总结任务

你是一个专家在多学科交叉领域的数据分析师。你擅长将图像处理、计算机视觉、信号处理、机器学习和深度学习技术融合应用，以解决复杂的问题。你精通帮助人们识别并理解特定社区或领域内的关系和结构，特别是在 '{domain_in_specific}' 等场景中。此外，你还擅长优化算法设计，如维特比算法(Viterbi algorithm)，以及在无监督学习、减少监督学习等领域进行创新研究。

您的任务是"给定一个实体名称和针对该实体的多个描述，将这些描述合并成一个简洁且全面的总结"。
严格按照以下格式要求操作。

## 输入格式
- entity_name: 需要总结描述的实体名称
- description_list: 针对该实体的多个描述组成的列表

## 输出格式要求(必须严格遵循)
输出必须是一段连贯的文字，用第三人称编写，包含实体的详细描述。

## 警告
- 总结必须包含所有描述中的关键信息
- 如果描述中存在矛盾，必须进行合理解决，提供连贯的总结
- 不要添加描述中没有的信息
- 确保用第三人称编写，并在描述中包含实体名称
- 尽可能丰富相关信息

## 处理要点
1. 仔细阅读所有的描述，提取关键信息
2. 解决可能存在的矛盾或不一致
3. 组织信息为流畅、连贯的段落
4. 确保总结全面涵盖实体的所有重要特征
5. 利用您在图像处理、计算机视觉、信号处理等领域的专业知识，提供深入的分析

## 当前任务
entity_name: {entity_name}
description_list: {description_list}

## 格式检查提示
处理完成后，请检查: 
1. 总结是否涵盖了所有描述中的关键信息
2. 是否解决了描述中可能存在的矛盾
3. 是否使用第三人称编写并包含实体名称
4. 文本是否流畅连贯
"""


PROMPTS[
    "e_r_continue_extraction"
] = """在上次提取中遗漏了许多实体或关系。使用相同的格式将它们添加到下面: 
"""

PROMPTS[
    "e_r_if_loop_extraction"
] = """似乎仍然遗漏了一些实体或关系。如果仍有实体或关系需要添加，请回答“yes|no”。
"""

PROMPTS[
    "e_continue_extraction"
] = """在上次提取中遗漏了许多实体。使用相同的格式将它们添加到下面: 
"""

PROMPTS[
    "e_if_loop_extraction"
] = """似乎仍然遗漏了一些实体。如果仍有实体或关系需要添加，请回答“yes|no”。
"""
PROMPTS[
    "r_continue_extraction"
] = """在上次提取中遗漏了许多关系。使用相同的格式将它们添加到下面: 
"""

PROMPTS[
    "r_if_loop_extraction"
] = """似乎仍然遗漏了一些关系。如果仍有关系需要添加，请回答“yes|no”。
"""

PROMPTS[
    "community_report"
] = """
# 社区报告生成任务

你是一个专家在多学科交叉领域的数据分析师。你擅长将图像处理、计算机视觉、信号处理、机器学习和深度学习技术融合应用，以解决复杂的问题。你精通帮助人们识别并理解特定社区或领域内的关系和结构，特别是在 '{domain_in_specific}' 等场景中。此外，你还擅长优化算法设计，如维特比算法(Viterbi algorithm)，以及在无监督学习、减少监督学习等领域进行创新研究。

## 任务描述
您的任务是"扮演 '{domain_in_specific}' 领域专家的角色，根据提供的文本内容，撰写社区的全面评估报告"。
严格按照以下格式要求操作。

## 输入格式
- input_text: 包含实体和关系信息的文本内容

## 输出格式要求(必须严格遵循)
输出必须是格式良好的JSON字符串，包含以下字段: 
- title: 社区名称，标题应简短但具体，最好能包含代表性命名实体
- summary: 社区整体结构的执行摘要，包括实体间关系和实体相关的重要信息
- rating: 一个介于0-10之间的浮点数，表示社区内实体带来的影响严重性。影响分数表示社区的重要性
- rating_explanation: 用一句话解释你给出的影响严重性评分rating的原因
- findings: 5-10个关键见解洞察，每个见解包含summary和explanation字段。每条洞察应包含一个简短的总结，并以多个段落的解释文本进行说明，请尽量详尽

输出格式必须严格按照以下JSON结构: 
    {{
        "title": <report_title>,
        "summary": <executive_summary>,
        "rating": <impact_severity_rating>,
        "rating_explanation": <rating_explanation>,
        "findings": [
            {{
                "summary":<insight_1_summary>,
                "explanation": <insight_1_explanation>
            }},
            {{
                "summary":<insight_2_summary>,
                "explanation": <insight_2_explanation>
            }}
        ]
    }}

## 警告
- 不要包含没有提供支持证据的信息
- 必须用中文完成回答
- 输出必须是有效的JSON格式
- 确保所有字符串值都用正确的双引号包裹
- 移除所有不必要的转义字符
- 结果应为单个JSON对象，可通过json.loads解析

## 报告要点
1. 标题应简短但具体，尽可能包含代表性命名实体
2. 摘要应概述社区整体结构，包括实体间关系和重要信息
3. 影响严重性评分应基于社区内实体所带来的影响
4. 详细发现应包含5-10个关键见解，每个见解包括简短总结和详细解释
5. 所有内容必须基于提供的文本证据，不得编造

## 示例
### 示例输入: 
input_text:

实体

id,entity,description
5,ABILA CITY PARK,Abila City Park 是 POK 集会的地点

关系

id,source,target,description
37,ABILA CITY PARK,POK RALLY,Abila City Park 是 POK 集会的地点
38,ABILA CITY PARK,POK,POK 在 Abila City Park 举办集会
39,ABILA CITY PARK,POKRALLY,POKRally 正在 Abila City Park 举行
40,ABILA CITY PARK,CENTRAL BULLETIN,Central Bulletin 正在报道 Abila City Park 上的 POK 集会

### 示例输出: 
{{
    "title": "Abila City Park 和 POK 集会",
    "summary": "社区围绕 Abila City Park 发展，该公园是 POK 集会的地点。该公园与 POK、POKRALLY 和 Central Bulletin 有关系，这些都与集会事件相关。",
    "rating": 5.0,
    "rating_explanation": "由于 POK 集会可能引发的动荡或冲突，影响评分为中等。",
    "findings": [
        {{
            "summary": "Abila City Park 作为核心地点",
            "explanation": "Abila City Park 是社区的核心实体，作为 POK 集会的地点。该公园是所有其他实体的共同联系点，表明它在社区中的重要性。公园与集会的关系可能导致公共秩序或冲突，具体取决于集会的性质及其引发的反应。"
        }},
        {{
            "summary": "POK 在社区中的角色",
            "explanation": "POK 是社区中的另一个关键实体，组织了在 Abila City Park 举行的集会。POK 的性质及其集会的目标可能是潜在威胁的来源，具体取决于其目标和引发的反应。理解 POK 和公园之间的关系对于理解社区动态至关重要。"
        }},
        {{
            "summary": "POKRALLY 作为重要事件",
            "explanation": "POKRALLY 是在 Abila City Park 举行的一个重要事件。这个事件是社区动态的一个关键因素，可能是潜在威胁的来源，具体取决于集会的性质及其引发的反应。集会与公园之间的关系对于理解社区的动态至关重要。"
        }},
        {{
            "summary": "Central Bulletin 的角色",
            "explanation": "Central Bulletin 正在报道 Abila City Park 上的 POK 集会。这表明该事件引起了媒体的关注，可能会放大其对社区的影响。Central Bulletin 的角色在塑造公众对事件及其相关实体的看法中可能具有重要意义。"
        }}
    ]
}}

## 当前任务
请使用以下文本来生成社区报告: 
{input_text}

## 格式检查提示
处理完成后，请检查: 
1. 输出是否为有效的JSON格式
2. 是否包含所有必需的字段(title, summary, rating, rating_explanation, findings)
3. rating是否为0-10之间的浮点数
4. findings是否包含5-10个见解，每个见解包含summary和explanation
5. 所有内容是否都基于提供的文本证据
"""


PROMPTS[
    "local_rag_response"
] = """
# 局部查询RAG回答任务

您是一位乐于助人的助手，可以回答有关所提供表格中数据的问题。
请务必用中文完成你的回答。

## 任务描述
您的任务是"根据提供的数据表信息，生成一个目标长度和格式的响应，回答用户的问题"。
严格按照以下格式要求操作。

## 输入格式
- context_data: 包含相关信息的数据表
- response_type: 期望的响应长度和格式

## 输出格式要求(必须严格遵循)
输出必须符合指定的response_type格式要求，并使用markdown标记样式。

数据引用：
如果答案的部分内容由特定数据支持，必须在相应位置标注数据来源，格式如下: 
"信息内容[数据来源: 数据来源类型(记录ids); 数据来源类型(记录ids)]"

## 警告
- 不要包含没有提供支持证据的信息
- 数据引用中不要列出超过5个记录ID，超过时使用"+more"表示
- 如果数据表中有直接支持问题的信息，直接返回相关部分而不要总结
- 如果不知道答案，直接说明不知道
- 不要输出你检查格式的过程

## 数据引用规则
1. 每个引用最多包含5个记录ID，超过时使用"+more"表示
2. 引用格式示例: "X是Y公司的所有者[数据来源: 来源(15, 16, 17, 18, 20, +more); 报告(1); 实体(5, 7); 关系(23)]"
3. 数据来源类型可以是"来源"、"报告"、"实体"或"关系"
4. 记录ID指的是相关数据记录的id(而非索引),同一个数据来源的不同id之间用逗号分隔
5. 多个数据来源之间用分号分隔

## 数据表
{context_data}

## 期望的响应格式
{response_type}

## 格式检查提示
处理完成后，请检查: 
1. 回答是否符合指定的response_type格式要求
2. 是否用正确格式标注了数据引用
3. 是否仅包含有支持证据的信息
4. 是否使用了markdown样式进行排版 
"""


PROMPTS[
    "global_map_rag_points"
] = """
# 全局查询RAG关键点提取任务

您是一位乐于助人的助手，可以回答有关所提供表格中数据的问题。
请务必用中文完成你的回答。

## 任务描述
您的任务是"根据提供的数据表信息，生成一个由能回答用户问题的关键点列表组成的响，总结输入数据表中的所有相关信息"。
严格按照以下格式要求操作。

## 输入格式
- context_data: 包含相关信息的数据表

## 输出格式要求(必须严格遵循)
输出必须是JSON格式，包含一个关键点列表，每个关键点包含: 
- description: 对该点的全面描述，带有数据来源引用
- score: 0-100之间的整数分数，表示该点在回答问题时的重要性

数据引用：
如果description的部分内容由特定数据支持，必须在相应位置标注数据来源，格式如下: 
"信息内容[数据来源: 数据来源类型(记录ids); 数据来源类型(记录ids)]"

总体来说，输出格式必须如下: 
{{
    "points": [
    {{"description": "点1的描述[数据来源: 报告 (report ids)]", "score": score_value}},
    {{"description": "点2的描述[数据来源: 报告 (report ids)]", "score": score_value}}
    ]
}}


## 警告
- 不要包含没有提供支持证据的信息
- 数据引用中不要列出超过5个记录ID，超过时使用"+more"表示
- 如果不知道答案或数据表中没有足够信息，直接说明
- 保留情态动词的原始含义和使用，如"或许"、"可能"或"将会"
- “我不知道”类型的回答score应该得0分。

## 数据引用规则
1. 每个引用最多包含5个记录ID，超过时使用"+more"表示
2. 引用格式示例: "X是Y公司的所有者[数据来源: 来源(15, 16, 17, 18, 20, +more); 报告(1); 实体(5, 7); 关系(23)]"
3. 数据来源类型可以是"来源"、"报告"、"实体"或"关系"
4. 记录ID指的是相关数据记录的id(而非索引),同一个数据来源的不同id之间用逗号分隔
5. 多个数据来源之间用分号分隔

## 数据表
{context_data}

## 格式检查提示
处理完成后，请检查: 
1. 输出是否为有效的JSON格式
2. 每个关键点是否包含description和score字段
3. 每个score是否为0-100之间的整数
4. 是否在description中用正确格式标注了数据引用
5. 是否仅包含有支持证据的信息
"""

PROMPTS[
    "global_reduce_rag_response"
] = """
# 全局查询RAG回答任务

您是一个乐于助人的助手，通过综合多个分析师的观点来回答有关数据集的问题。
请务必用中文完成你的回答。

## 任务描述
您的任务是"根据多个专注于数据集不同部分的分析师报告，生成一个全面的综合回答"。
严格按照以下格式要求操作。

## 输入格式
- report_data: 多个分析师的报告，**按重要性降序排列**
- response_type: 期望的响应长度和格式

## 输出格式要求(必须严格遵循)
输出必须符合指定的response_type格式要求，并使用markdown标记样式。
最终回答应删除分析师报告中所有不相关的信息，合并为一个全面的答案。
必须保留分析师报告中的所有数据引用，格式如下: 
"信息内容[数据来源: 数据来源类型(记录ids); 数据来源类型(记录ids)]"

## 警告
- 不要包含没有提供支持证据的信息
- 数据引用中不要列出超过5个记录ID，超过时使用"+more"表示
- 如果不知道答案或报告中没有足够信息，直接说明
- 保留情态动词的原始含义和使用，如"或许"、"可能"或"将会"
- 不要提及多个分析师在分析过程中的作用
- 不要输出你检查格式的过程


## 数据引用规则
1. 每个引用最多包含5个记录ID，超过时使用"+more"表示
2. 引用格式示例: "X是Y公司的所有者[数据来源: 来源(15, 16, 17, 18, 20, +more); 报告(1); 实体(5, 7); 关系(23)]"
3. 数据来源类型可以是"来源"、"报告"、"实体"或"关系"
4. 记录ID指的是相关数据记录的id(而非索引),同一个数据来源的不同id之间用逗号分隔
5. 多个数据来源之间用分号分隔


## 分析师报告(按重要性降序排列)
{report_data}

## 期望的响应格式
{response_type}

## 格式检查提示
处理完成后，请检查: 
1. 回答是否符合指定的response_type格式要求
2. 是否用正确格式标注了数据引用
3. 是否删除了所有不相关信息
4. 是否保留了情态动词的原始含义
5. 是否使用了markdown样式进行排版
6. 是否没有提及分析师在分析过程中的作用
"""

PROMPTS[
    "naive_rag_response"
] = """
# 基础知识检索回答任务

您是一位乐于助人的助手，可以根据提供的知识回答用户问题。

## 任务描述
您的任务是"根据提供的知识内容，生成一个目标长度和格式的响应，回答用户的问题"。
严格按照以下格式要求操作。

## 输入格式
- content_data: 您掌握的相关知识内容
- response_type: 期望的响应长度和格式

## 输出格式要求(必须严格遵循)
输出必须符合指定的response_type格式要求。
回答应总结提供的知识内容中的相关信息，可适当加入相关的通用知识。

## 警告
- 不要编造任何内容
- 不要包含没有提供支持证据的信息
- 如果不知道答案或提供的知识不包含足够信息，请直接说明

## 您掌握的知识
{content_data}

## 期望的响应格式
{response_type}

## 格式检查提示
处理完成后，请检查: 
1. 回答是否符合指定的response_type格式要求
2. 是否只基于提供的知识内容回答问题
3. 是否没有编造任何内容
"""



PROMPTS[
    'table_parse_entities_and_relationships'
] = '''
# 表格实体与关系提取任务

您的任务是"给定一个可能与此活动相关的表格文本、表格标题、表格概述和一个实体类型列表，从表格中的数据行识别所有这些类型的实体以及这些实体之间的所有关系"。
严格按照以下格式要求操作。

## 输入格式
- entity_types: 实体类型列表
- table_title: 表格标题
- table_summary: 表格概述
- table_content: 表格预览内容
- table_row: 待处理的表格行

## 注意事项
当表格里出现整行/整列的省略号(...)时表明后续的内容因为篇幅原因被省略了。
通常来说，表格的每个单元格不会包含超过一个实体。

## 输出格式要求(必须严格遵循)

1. 每个实体必须使用确切的格式: 
   ("entity"{tuple_delimiter}实体名称{tuple_delimiter}实体类型{tuple_delimiter}实体描述){record_delimiter}
   
2. 每个关系必须使用确切的格式: 
   ("relationship"{tuple_delimiter}源实体名称{tuple_delimiter}目标实体名称{tuple_delimiter}关系描述{tuple_delimiter}关系强度数字){record_delimiter}

3. 所有条目之间必须使用 {record_delimiter} 作为分隔符，每个条目后面都跟随 {record_delimiter}

4. 完成后必须添加 {completion_delimiter} 标记

## 警告

- 不要使用任何其他格式或添加额外内容
- 实体类型必须从提供的列表中选择
- 实体描述指: 实体属性和活动的详细描述
- 关系描述指: 解释为什么认为源实体和目标实体彼此相关
- 关系强度数字必须是1-10之间的整数，表示源实体和目标实体之间关系的强度，分数值越大表示强度越高

## 示例
### 示例1
#### 输入示例: 
entity_types: [数据类型, 数据取值, 数据精度, 数据版本, 频段规约]
table_title: 表B-70. 类型1 伪距校正信息报文
table_summary: 该表格展示了GBAS类型1伪距校正信息报文的编码格式，包括数据内容、比特数、取值范围和精度。
table_content: 
| 数据内容     | 比特数 | 取值范围       | 精度  |
|--------------|--------|----------------|-------|
| 校正Z计数    | 14     | 0 to 1199.9 s | 0.1 s |
| 附加电文标记 | 2      | 0 to 3        | 1     |
| 测量数 (N)   | 5      | 0 to 18        | 1     |
| 测量类型     | 3      | 0 to 7         | 1     |
| ...          | ...    | ...            | ...   |
table_row: 
| 数据内容     | 比特数 | 取值范围       | 精度  |
|--------------|--------|----------------|-------|
| 附加电文标记 | 2      | 0 to 3        | 1     |
| 测量数 (N)   | 5      | 0 to 18        | 1     |

#### 正确输出示例: 
("entity"{tuple_delimiter}"GBAS伪距校正信息报文-附加电文标记"{tuple_delimiter}"数据类型"{tuple_delimiter}"GBAS 伪距校正信息报文数据取值的一种"){record_delimiter}
("entity"{tuple_delimiter}"GBAS伪距校正信息报文-测量数"{tuple_delimiter}"数据类型"{tuple_delimiter}"GBAS 伪距校正信息报文数据取值的一种"){record_delimiter}
{completion_delimiter}

### 示例2
#### 输入示例: 
entity_types: [数据类型, 数据取值, 数据精度, 数据版本, 频段规约]
table_title: 表D-6 GBAS 频道分配示例  
table_summary: 该表格展示了GBAS频道编号、频率与RPDS/RSDS的对应关系。
table_content: 
| 频道编码 (N) | 频率(F) (MHz) | 参考通道数据选择器(RPDS) 或参考站数据选择器(RSDS)        | 精度  |
|--------------|----------------|------------------------------------------------------|-------|
| 20001        | 108. 025       | 0                                                    | 0.1 s |
| 20002        | 108. 05        | 0                                                    | 1     |
| 20003        | 108.075        | 0                                                    | 1     |
| ...          | ...            | ...                                                  | ...   |
table_row: 
| 频道编码 (N) | 频率(F) (MHz) | 参考通道数据选择器(RPDS) 或参考站数据选择器(RSDS)        | 精度  |
|--------------|----------------|------------------------------------------------------|-------|
| 20001        | 108. 025       | 0                                                    | 0.1 s |

#### 正确输出示例: 
("entity"{tuple_delimiter}"GBAS频道编码(N)-20001"{tuple_delimiter}"频段规约"{tuple_delimiter}"GBAS 频道分配中的一个频道"){record_delimiter}
("entity"{tuple_delimiter}"108. 025MHz频率"{tuple_delimiter}"频段规约"{tuple_delimiter}"无线通信频率"){record_delimiter}
("relationship"{tuple_delimiter}"GBAS频道编码(N)-20001"{tuple_delimiter}"108. 025MHz频率"{tuple_delimiter}"描述了对应频率频段的用途"{tuple_delimiter}8){record_delimiter}
{completion_delimiter}

## 当前任务
entity_types: {entity_types}
table_title: {table_title}
table_summary: {table_summary}
table_content: 
{table_content}
table_row: 
{table_row}

## 格式检查提示
处理完成后，请检查: 
1. 所有实体格式是否为 ("entity"{tuple_delimiter}实体名称{tuple_delimiter}实体类型{tuple_delimiter}实体描述){record_delimiter}
2. 所有关系格式是否为 ("relationship"{tuple_delimiter}源实体名称{tuple_delimiter}目标实体名称{tuple_delimiter}关系描述{tuple_delimiter}关系强度数字){record_delimiter}
3. 实体类型是否从提供的entity_types列表中选择
4. 末尾是否添加了 {completion_delimiter}
'''

PROMPTS[
    'table_new_title'
] = '''
# 表格标题重写任务

您的任务是"给定一个表格的标题、部分内容预览和表格的上下文，根据上下文重写表格的标题"。
严格按照以下格式要求操作。

## 输入格式
- table_title: 原始表格标题
- table_content: 表格内容预览
- context: 表格的上下文信息

## 注意事项
当表格里出现整行/整列的省略号(...)时表明后续的内容因为篇幅原因被省略了。

## 输出格式要求(必须严格遵循)
输出必须使用确切的格式: 
("table_new_title"{tuple_delimiter}重写后的表格标题)

## 警告
- 不要使用任何其他格式或添加额外内容
- 只重写标题名称部分，保留表格编号不变，用英文冒号":"分隔表格编号和标题名称
- 如果无法理解表格内容或无法按要求重写标题，请直接输出: ("table_new_title"{tuple_delimiter}"None")

## 重写要点
1. 阅读表格的原始标题，理解标题所说明的表格用途
2. 阅读表格的表头和内容，思考标题与表头信息是否明确了表格聚焦的对象
3. 对象应该精准地限定在一个实体上，例如将"表XX-各数据内容的位数及编码"改为"表XX:某某协议各数据内容的位数及编码"
4. 确保新标题准确描述了表格的内容和用途

## 示例
### 示例1
#### 输入示例: 
table_title: 表D-6 频道分配示例
table_content:
| 频道编码 (N) 	| 频率(F) (MHz) 	| 参考通道数据选择器 或参考站数据选择器 (RPDS) (RSDS) 	|
|--------------	|----------------	|------------------------------------------------------	|
| 20001        	| 108. 025       	| 0                                                    	|
| 20002        	| 108. 05        	| 0                                                    	|
| 20003        	| 108.075        	| 0                                                    	|
| ...          	| ...            	| ...                                                  	|

context: 
...

7.8 参考路径数据和参考站数据选择器
映射方案为每个GBAS进近提供了一个独有的频道编号。该频道编号由5个数字字符组成，范围是从20001到39999。这个频道编号能使GBAS机载子系统以正确的频率选择最终进近段(FAS)数据段，来确定进近方式。作为4类信息FAS定义数据的一部分，正确的FAS数据段可根据参考路径选择器(RPDS)来选择。表D - 6显示的是频道数量，频率和RPDS之间关系。基于参考站数据选择器(RSDS)的布局方案同样适用于定位服务的选择。RSDS在类型2报文中广播，并允许选择一个特有的GBAS的地面子系统提供定位服务。对于不提供定位服务和广播额外的星历数据的GBAS地面子系统，RSDS用255进行编码。所有由地面子系统广播的RPDS和RSDS，必须在信号范围内有唯一的无线信号频率。RSDS值不能与任意一个广播的RPDS值相同。

7.9 服务提供商的RPDS 和RSDS 分配
RPDS 和RSDS 的分配要加以控制，以避免在数据广播频率的保护区域内的频道数重复使用。因此，对于某一GBAS的地面子系统，一个给定的无线电频率范围内，GBAS 服务提供商必须确保RPDS 和RSDS 只分配一次。RPDS 和RSDS 的分配与VHF 数据广播频率和时段的分配要同时进行。

7.10 GBAS 标识
GBAS 标识(ID)用于唯一地识别GBAS 地面子系统，并由一个给定的频率在GBAS 的覆盖范围内广播。飞机将使用GBAS 的地面子系统(由特定的GBAS 标识识别)中一个或多个广播站广播的数据进行导航。

7.11 最后进近段(FAS)路径

7.11.1 最后进近段(FAS)路径是空间的一条线。这条线由着陆入口点/虚拟入口点(LTP/FTP)，飞行路径对齐点(FPAP)，穿越入口高度(TCH)和下滑道角度(GPA)定义。这些参数由4 类信息的FAS 数据块或机载数据库提供的数据决定。这些参数和FAS 路径的关系在图D-6 中加以说明。

ATT-D-28

表D-6 频道分配示例
【当前表格所处位置】

7.11.1.1 SBAS 和一些GBAS 进近的FAS 数据块存放在支持一个SBAS 和GBAS 的共用机载数据库中。当4 类信息不能广播时，国家负责提供FAS 数据以支持APV 程序。这些数据由包含在FAS 数据块的参数、RSDS 和相关的广播频率组成。对特定的进近程序，FAS 数据块在附录B，3.6.4.5.1 和表B-66 中描述。
...

#### 正确输出示例: 
("table_new_title"{tuple_delimiter}"表D-6:GBAS频道编号、频率与RPDS/RSDS的对应关系")

## 当前任务
table_title: {table_title}
table_content: {table_content}
context: {context}

## 格式检查提示
处理完成后，请检查: 
1. 输出格式是否为 ("table_new_title"{tuple_delimiter}重写后的表格标题)
2. 是否保留了原表格编号并使用英文冒号":"分隔编号和标题名称
3. 重写的标题是否准确描述了表格内容并基于上下文进行了明确限定
'''

PROMPTS[
    'table_parse_summary'
] = '''
# 表格内容一句话总结任务

您的任务是"给定表格的部分内容以及表格的标题，输出一句话形式的总结，简要阐述该表格的整体信息、描述的各实体之间的关系语义等"。
严格按照以下格式要求操作。

## 输入格式
- table_title: 表格标题
- table_content: 表格内容预览

## 注意事项
当表格里出现整行/整列的省略号(...)时表明后续的内容因为篇幅原因被省略了。

## 输出格式要求(必须严格遵循)
输出必须使用确切的格式: 
("table_summary"{tuple_delimiter}表格一句话总结)

## 警告
- 不要使用任何其他格式或添加额外内容
- 总结必须是一句完整的话，简明扼要地概括表格主要内容
- 如果无法理解表格内容或无法按要求总结，请直接输出: ("table_summary"{tuple_delimiter}"None")

## 总结要点
1. 理解表格的标题和内容之间的关系
2. 识别表格中展示的主要实体和它们之间的关系
3. 总结表格要传达的核心信息
4. 将总结压缩为一句话的形式

## 示例
### 示例1
#### 输入示例: 
table_title: "CAT I 侧向告警限"
table_content: 
| 飞机沿着最终进近路径距离LTP/FTP的水平距离 | 横向告警限 (单位: m)        |
|-------------------------------------------|-----------------------------|
| 291                                       | FASLAL                      |
| 873                                       | 0.0044D (m) + FASLAL - 3.85 |
| D > 7500                                 | FASLAL + 29.15              |

#### 正确输出示例: 
("table_summary"{tuple_delimiter}"该表格显示了飞机沿着最终进近路径距离LTP/FTP的水平距离与横向告警限之间的关系，其中部分横线告警限通过 FASLAL 值进行定义。")

### 示例2
#### 输入示例: 
table_title: "表 B-71A. 类型 2 GBAS 相关报文"
table_content: 
| 数据内容     | 比特数 | 取值范围       | 精度  |
|--------------|--------|----------------|-------|
| 校正Z计数    | 14     | 0 to 1199.9 s | 0.1 s |
| 附加电文标记 | 2      | 0 to 3           | 1     |
| 测量数 (N)   | 5      | 0 to 18        | 1     |
| 测量类型     | 3      | 0 to 7           | 1     |
| ...          | ...    | ...            | ...   |

#### 正确输出示例: 
("table_summary"{tuple_delimiter}"该表格描述了GBAS相关报文在不同数据内容情形下占用的比特数、取值范围及精度。")

## 当前任务
table_title: "{table_title}"
table_content: 
{table_content}

## 格式检查提示
处理完成后，请检查: 
1. 输出格式是否为 ("table_summary"{tuple_delimiter}表格一句话总结)
2. 总结是否只有一句话，且能简明扼要地概括表格主要内容
3. 总结是否概括了表格中描述的主要实体和关系
'''



PROMPTS[
    "get_entity_type_and_description"
] = """
# 实体类型与描述提取任务

您的任务是"给定一个实体名称entity_name、可能与此活动相关的文本文件text和一个实体类型列表entity_types，提取出实体的类型和描述"。
严格按照以下格式要求操作。

## 输入格式
- entity_name: 实体名称
- entity_types: 实体类型列表
- text: 需要分析的文本内容

## 输出格式要求(必须严格遵循)
输出必须是JSON字典格式，包含以下字段: 
- entity_name: 实体名称，首字母大写
- entity_type: 从提供的实体类型列表中选择的类型
- entity_description: 实体属性和实体详细描述

## 警告
- 不要输出任何额外的文本或标记
- 实体类型必须从提供的列表中选择
- 如果文本中没有直接定义实体，可以通过上下文推断其类型和描述
- 输出必须是有效的JSON格式

## 示例
### 示例1
#### 输入示例: 
entity_name: 高斯平滑
entity_types: [数据预处理方法, 图像处理结果, 数据结构, 图像处理方法]
text:
骨架化与谱线分离方法。

1. 为了避免曲线"毛刺"对骨架提取的不利影响，可先进行高斯平滑处理，并设置合适阈值将其转化为二值图像，在此基础上再提取曲线骨架。通过提取 m-D 曲线的骨架，就可达到抑制距离像旁瓣引起的曲线变"粗"的现象。
2. 根据形态学图像处理理论，骨架提取可由"腐蚀"运算和"开"运算实现。抑制 m-D 曲线的距离像旁瓣引起的曲线变"粗"的现象，提取曲线骨架。

#### 正确输出示例: 
{{
    "entity_name": "高斯平滑",
    "entity_type": "数据预处理方法",
    "entity_description": "用于减少噪声，提高骨架提取的准确性"
}}

### 示例2
#### 输入示例: 
entity_name: 步态身份识别
entity_types: [算法, 人物, 活动, 问题描述]
text:
# 华为六月沟通小结

标签: 孙泽钰
日期: June 21, 2023

- 目前来看步态身份识别基本不用做进一步的优化实验

## 后续两个方面的工作: 

### 专利申请(陈重)

- **他们对初稿的意见: **
    1. 系统太大(可以不把雷达部分加进来)
        - 理由是涵盖太大反而不好保护，应该针对其中关键的算法节点进行保护；
        - **另一种思路(陈重提的): 可以涵盖整个系统，然后同时提2-3个

#### 正确输出示例: 
{{
    "entity_name": "步态身份识别",
    "entity_type": "算法",
    "entity_description": "一种生物识别技术，用于根据人的步态识别其身份"
}}

## 当前任务
entity_name: {entity_name}
entity_types: {entity_types}
text: {input_text}

## 格式检查提示
处理完成后，请检查: 
1. 输出是否为有效的JSON格式
2. 实体类型是否从提供的entity_types列表中选择
3. 是否包含所有必需的字段(entity_name, entity_type, entity_description)
"""


PROMPTS[
    "get_relationship_description_and_strength"
] = """
# 实体关系与强度提取任务

您的任务是"给定一个可能与此活动相关的文本文件和两个实体，从文本中识别这两个实体之间的关系及其强度数字"。
严格按照以下格式要求操作。

## 输入格式
- source_entity: 源实体名称
- target_entity: 目标实体名称
- text: 需要分析的文本内容

## 输出格式要求(必须严格遵循)
输出必须是JSON字典格式，包含以下字段: 
- source_entity: 源实体的名称
- target_entity: 目标实体的名称
- relationship_description: 解释为什么认为源实体和目标实体彼此相关
- relationship_strength: 一个介于1到10之间的整数，表示源实体和目标实体之间关系的强度

## 警告
- 不要输出任何额外的文本或标记
- 关系强度数字必须是1-10之间的整数，分数值越大表示强度越高
- 如果文本中没有直接定义实体间关系，可以通过上下文推断
- 输出必须是有效的JSON格式

## 示例
### 示例1
#### 输入示例: 
source_entity: 高斯平滑
target_entity: 二值图像
text:
骨架化与谱线分离方法。

1. 为了避免曲线"毛刺"对骨架提取的不利影响，可先进行高斯平滑处理，并设置合适阈值将其转化为二值图像，在此基础上再提取曲线骨架。通过提取 m-D 曲线的骨架，就可达到抑制距离像旁瓣引起的曲线变"粗"的现象。
2. 根据形态学图像处理理论，骨架提取可由"腐蚀"运算和"开"运算实现。抑制 m-D 曲线的距离像旁瓣引起的曲线变"粗"的现象，提取曲线骨架。

#### 正确输出示例: 
{{"source_entity": "高斯平滑", "target_entity": "二值图像", "relationship_description": "为曲线骨架提取提供更清晰的输入", "relationship_strength": 8}}

### 示例2
#### 输入示例: 
source_entity: 陈重
target_entity: 初稿
text:
# 华为六月沟通小结

标签: 孙泽钰
日期: June 21, 2023

- 目前来看步态身份识别基本不用做进一步的优化实验

## 后续两个方面的工作: 

### 专利申请(陈重)

- **他们对初稿的意见: **
    1. 系统太大(可以不把雷达部分加进来)
        - 理由是涵盖太大反而不好保护，应该针对其中关键的算法节点进行保护；
        - **另一种思路(陈重提的): 可以涵盖整个系统，然后同时提2-3个

#### 正确输出示例: 
{{"source_entity": "陈重", "target_entity": "初稿", "relationship_description": "陈重对专利申请初稿提供了改进建议", "relationship_strength": 9 }}

## 当前任务
source_entity: {source_entity}
target_entity: {target_entity}
text: {input_text}

## 格式检查提示
处理完成后，请检查: 
1. 输出是否为有效的JSON格式
2. 关系强度数字是否为1-10之间的整数
3. 是否包含所有必需的字段(source_entity, target_entity, relationship_description, relationship_strength)
"""
