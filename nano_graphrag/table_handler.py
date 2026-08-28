import re
import os
import pandas as pd
import pickle
import json
from tabulate import tabulate
from dataclasses import dataclass, asdict
from io import StringIO
from typing import List, Optional, Dict, Tuple
from .prompt import PROMPTS
from ._utils import (
    logger,
    compute_mdhash_id,
    split_string_by_multi_markers,
    clean_str,
)


@dataclass
class TableHandler:
    html_table: Optional[str] = None
    list_table: Optional[List[List[str]]] = None
    markdown_table: Optional[str] = None
    title_name: Optional[str] = None
    llm_title_name: Optional[str] = None
    key: Optional[str] = None
    splitter_index: int = 0
    summary: Optional[str] = None
    pre_content: Optional[str] = None
    post_content: Optional[str] = None
    global_config: Optional[Dict] = None

    def __post_init__(self):
        pass

    def load_html_table(self, html_table: str) -> None:
        self.html_table = html_table
        self.to_markdown()
        self.get_hash()

    async def set_context(self, pre_content: str, post_content: str) -> None:
        self.pre_content = pre_content
        self.post_content = post_content  
        await self.search_title_from_neighbor_contents()
        await self.generate_new_title()
        await self.parse_summary()

    def to_markdown(self) -> str:
        if self.markdown_table is not None:
            return self.markdown_table
        
        # 获取列名并构造 converters 字典，将所有列转换为字符串。这里实测发现df.applymap(str)不起作用，只能这样写。
        df = pd.read_html(StringIO(self.html_table))[0]
        converters = {column: str for column in df.columns}
        df = pd.read_html(StringIO(self.html_table), converters=converters)[0]  # 读取表格
        # 转换为Markdown格式
        lines = tabulate(df, headers='keys', tablefmt='github', showindex=False).splitlines()
        # 正则表达式匹配过滤所有单元格值为 'nan' 的行
        nan_row_pattern = re.compile(r'^\s*\|(\s*nan\s*\|)+\s*$')
        filtered_lines = [line for line in lines if not nan_row_pattern.match(line)]
        # 正则表达式匹配找到表格分隔符行的索引
        separator_pattern = re.compile(r'^\s*\|?(\s*-+\s*\|)+\s*$')
        self.splitter_index = [i for i,line in enumerate(filtered_lines) if separator_pattern.match(line)][0]
        # 将过滤后的行重新组合为 Markdown 表格字符串
        self.markdown_table = "\n".join(filtered_lines)
        self.list_table = [row.split('|') for row in filtered_lines]    # 包含每行两侧的空字符
        return self.markdown_table

    def get_hash(self) -> str:
        if self.key is None:
            # 注意！这里表格的hash id是根据其markdown格式文本计算的，并不是原始chunk文本！
            self.key = compute_mdhash_id(self.to_markdown(), prefix='chunk-')
        return self.key 

    def iter_markdown(self, header=False, start=0, end=None, step=1, overlap=0):
        if self.list_table is None:
            raise ValueError('List table not loaded')
        header_cache = '\n'.join(
                [
                    "|".join(self.list_table[i])
                    for i in range(self.splitter_index + 1)
                ]
            ) + '\n'
        start_index = self.splitter_index + start + 1
        end_index = len(self.list_table) if end is None else min(self.splitter_index + end + 2, len(self.list_table))   # 上限，不会取到
        if start_index >= end_index:
            raise ValueError("Invalid range: start index must be less than end index.")        
        
        i = start_index
        while i < end_index:
            current_markdown = header_cache if header else ''
            rows = self.list_table[i:i+step]
            # additional_index用于记录因为表格中此行第一个非空单元格是合并单元格而额外增多的行数
            # 我们需要将这些行一并加入到当前的markdown中、在一次解析中处理
            additional_index = 0        

            while i + step + additional_index < end_index and self.list_table[i + step + additional_index][1] == rows[-1][1]:
                rows.append(self.list_table[i + step + additional_index])
                additional_index += 1

            i += max(1, step + additional_index - overlap)  # 确保不会陷入死循环
            current_markdown += '\n'.join(
                [
                    "|".join(row)
                    for row in rows
                ]
            ) + '\n'

            yield current_markdown

    def __sizeof__(self):
        if self.list_table is None:
            return 0
        return len(self.list_table) - self.splitter_index - 1

    def _row_to_html(self, row: List[str]) -> str:
        html = '  <tr>\n'
        for cell in row:
            html += f'    <td>{cell}</td>\n'
        html += '  </tr>'
        return html

    def save_th(self):
        os.makedirs(os.path.join(self.global_config["working_dir"], 'table_handlers'), exist_ok=True)
        with open(os.path.join(self.global_config["working_dir"], 'table_handlers', f'{self.get_hash()}.json'), 'w', encoding='utf-8') as f:
            f.write(
                json.dumps(
                    {k: v for k, v in asdict(self).items() if v is not None and k != 'global_config'}, 
                    ensure_ascii=False, 
                    indent=4,
                )
            )

    async def search_title_from_neighbor_contents(self) -> None:
        pattern = re.compile(r'^[#\s]*表[^\n]+$', re.MULTILINE)
        maybe_titles = []

        if self.pre_content:
            matches = list(pattern.finditer(self.pre_content))
            if matches:
                # 找到匹配的位置和到当前table的距离
                maybe_titles.append(
                    {
                        "source": -1,   # -1表示pre_content
                        "place_to_table": matches[-1].end() - len(self.pre_content),
                        "content": matches[-1].group(0),
                    }
                )
        
        if self.post_content:
            matches = list(pattern.finditer(self.post_content))
            if matches:
                # 找到匹配的位置和到当前table的距离
                maybe_titles.append(
                    {
                        "source": 1,   # 1表示post_content
                        "place_to_table": matches[0].start(),
                        "content": matches[0].group(0),
                    }
                )

        if not len(maybe_titles):
            self.title_name = None        
        else:
            # maybe_titles按照到当前table的距离("place_to_table"的绝对值)从小到大排序，
            # 如果距离相同，优先选择来自pre_content（即source == -1）
            self.title_name = sorted(maybe_titles, key=lambda x: (abs(x["place_to_table"]), x["source"]))[0]["content"].strip(' #\n').strip()
        return self.title_name

    async def parse_summary(self) -> str:
        if self.summary is not None:
            return self.summary        
        if self.list_table is None or self.html_table is None:
            raise ValueError('No table loaded')

        use_llm_func: callable = self.global_config['best_model_func']

        summary_prompt = PROMPTS['table_parse_summary'].format(
            table_content=self.to_markdown(), 
            table_title=self.llm_title_name if self.llm_title_name is not None else self.title_name,
            tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"]
        )

        retry_time = 0
        summary = None
        while summary is None and retry_time < 3:
            try:
                summary_str: str = await use_llm_func(prompt=summary_prompt)
                match = re.search(r"\((.*)\)", summary_str.replace('（','(').replace('）',')'))
                if match is None:  # 没提取到指定格式的内容，跳过
                    raise ValueError('No summary extracted')
                summary = clean_str(split_string_by_multi_markers(
                    match.group(1), [PROMPTS["DEFAULT_TUPLE_DELIMITER"]]
                )[-1]).strip('"').strip("'").strip()
                assert summary is not None and len(summary) > 0 and summary.lower() != 'none'
            except Exception as e:
                summary = None
            finally:
                retry_time += 1
        self.summary = summary
        return summary

    async def parse_entities(
        self, step: int = 1, overlap: int = 0
    ) -> Tuple[Dict, Dict]:
        from ._extracter import (
            extract_format_matcher,
        )
        if self.list_table is None or self.html_table is None:
            raise ValueError('No table loaded')

        use_llm_func: callable = self.global_config['best_model_func']

        context_base = dict(
            tuple_delimiter=PROMPTS['DEFAULT_TUPLE_DELIMITER'],
            record_delimiter=PROMPTS['DEFAULT_RECORD_DELIMITER'],
            completion_delimiter=PROMPTS['DEFAULT_COMPLETION_DELIMITER'],
            entity_types=','.join(PROMPTS['DEFAULT_ENTITY_TYPES']),
        )

        final_result = ''
        for md_row_with_header in self.iter_markdown(header=True, step=step, overlap=overlap):
            hint_prompt=PROMPTS['table_parse_entities_and_relationships'].format(
                table_content=self.to_markdown(),
                table_row=md_row_with_header,
                table_title=self.llm_title_name if self.llm_title_name is not None else self.title_name,
                table_summary=self.summary,
                **context_base,
            )
            extraction_result = await use_llm_func(
                prompt=hint_prompt
            )
            final_result += extraction_result
        result = await extract_format_matcher(final_result, self.get_hash(), context_base)
        self.extract_result = result
        self.save_th()

        # chunk_key = self.get_hash()
        # existing_data = {}
        # os.makedirs('qa', exist_ok=True)
        # if os.path.exists(f'qa/{chunk_key}.pkl'):
        #     with open(f'qa/{chunk_key}.pkl', 'rb') as file:
        #         existing_data = pickle.load(file)
        #         assert existing_data['chunk_key'] == chunk_key, f"Chunk key mismatch: {existing_data['chunk_key']} != {chunk_key}"

        # existing_data['type'] = 'table'
        # existing_data['chunk_key'] = chunk_key
        # existing_data["e_r_extraction"] = {
        #     'prompt': PROMPTS['table_parse_entities_and_relationships'].format(
        #                 table_content=self.to_markdown(),
        #                 table_row=self.to_markdown(),
        #                 table_title=self.llm_title_name if self.llm_title_name is not None else self.title_name,
        #                 table_summary=self.summary,
        #                 **context_base,
        #             ),
        #     'qwen2.5-72b': {'response': final_result, 'result': result},
        # }
        # with open(f'qa/{chunk_key}.pkl', 'wb') as file:
        #     pickle.dump(existing_data, file)


        return result

    async def generate_new_title(self) -> str:
        if self.list_table is None or self.html_table is None:
            raise ValueError('No table loaded')
        
        context = '\n...\n'
        if self.pre_content:
            context += self.pre_content
        context += '\n\n【当前表格位置】\n\n'
        if self.post_content:
            context += self.post_content + '\n...\n'
        if context is None:
            logger.warning('No context added')

        use_llm_func: callable = self.global_config['best_model_func']

        new_title_prompt = PROMPTS['table_new_title'].format(
            table_content=self.to_markdown(),
            table_title=self.title_name,
            context=context,
            tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"]
        )
        retry_time = 0
        new_title = None
        while new_title is None and retry_time < 3:
            try:
                new_title_str: str = await use_llm_func(prompt=new_title_prompt)
                match = re.search(r"\((.*)\)", new_title_str.replace('（','(').replace('）',')'))
                if match is None:  # 没提取到指定格式的内容，跳过
                    raise ValueError('No title extracted')
                new_title = clean_str(split_string_by_multi_markers(
                    match.group(1), [PROMPTS["DEFAULT_TUPLE_DELIMITER"]]
                )[-1]).strip('"').strip("'").strip()
                assert new_title is not None and len(new_title) > 0 and new_title.lower() != 'none' and len(new_title.split(':')) == 2
            except Exception as e:
                new_title = None
            finally:
                retry_time += 1
        self.llm_title_name = new_title
        return self.llm_title_name
