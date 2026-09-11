#!/usr/bin/env python3
"""从 LangSmith 提取数据并保存为 JSONL 文件,数据集可配置。"""

import os
import json
import argparse
from langsmith import Client
from dotenv import load_dotenv

load_dotenv()


def extract_langsmith_data(project_name, model_name, dataset_name, api_key):
    """从 LangSmith 提取数据并保存为 JSONL 文件。"""
    print(f"正在从 LangSmith 项目提取数据:{project_name}")
    print(f"使用的数据集:{dataset_name}")
    
    client = Client(api_key=api_key)
    
    # 读取项目以获取参考数据集 ID
    project_data = client.read_project(project_name=project_name)
    
    # 读取参考数据集以获取示例
    examples_gen = client.list_examples(dataset_id=project_data.reference_dataset_id)
    examples = []
    for example in examples_gen:
        examples.append(example)
    examples_dict = {example.id: example for example in examples}
    
    # 读取项目运行记录,以获取各次运行的输入、输出以及 reference_example_ids
    output_runs = client.list_runs(
        project_name=project_name,
        is_root=True
    )
    
    runs = []
    for run in output_runs:
        if run.outputs is not None and run.outputs.get("final_report") is not None:
            runs.append(run)
    
    output_jsonl = [
        {
            "id": examples_dict[run.reference_example_id].metadata["id"],
            "prompt": run.inputs["inputs"]["messages"][0]["content"],
            "article": run.outputs["final_report"],
        } for run in runs
    ]
    
    # 将 output_jsonl 写入 tests/expt_results 目录下的 JSONL 文件
    output_file_path = f"tests/expt_results/{dataset_name}_{model_name}.jsonl"
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    with open(output_file_path, 'w', encoding='utf-8') as f:
        for item in output_jsonl:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"数据已写入 {output_file_path}")
    print(f"记录总数:{len(output_jsonl)}")
    return output_file_path


def main():
    parser = argparse.ArgumentParser(description='从 LangSmith 项目提取数据')
    parser.add_argument('--project-name', required=True, help='LangSmith 项目名称')
    parser.add_argument('--model-name', required=True, help='用于输出文件名的模型名称')
    parser.add_argument('--dataset-name', required=True, help='用于输出文件名的数据集名称')
    parser.add_argument('--api-key', help='LangSmith API 密钥(默认使用 LANGSMITH_API_KEY 环境变量)')
    
    args = parser.parse_args()
    
    api_key = args.api_key or os.getenv('LANGSMITH_API_KEY')
    if not api_key:
        raise ValueError("必须通过 --api-key 参数或 LANGSMITH_API_KEY 环境变量提供 API 密钥")
    
    extract_langsmith_data(
        project_name=args.project_name,
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        api_key=api_key
    )


if __name__ == "__main__":
    main()