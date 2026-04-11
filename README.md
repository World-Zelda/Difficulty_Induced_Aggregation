# Math Inference Evaluation Toolkit

This is a toolkit for evaluating the performance of large language models (LLMs) on mathematical reasoning tasks. The project uses vLLM for efficient inference and supports multi-sample generation and evaluation on datasets like MATH-500.

## Project Structure

```
.
├── inference.py                 # Main inference script, uses vLLM to generate model outputs
├── eval/                        # Evaluation scripts directory
│   ├── ats.py                   # ATS evaluation method
│   ├── D-induce.py              # D-induce evaluation method
│   ├── dts.py                   # DTS evaluation method
│   ├── majority_vote.py         # Majority voting evaluation method
│   └── rs.py                    # RS evaluation method
├── scripts/                     # Scripts directory
│   └── run.sh                   # Bash script for batch running experiments
└── utils/                       # Utilities directory
    └── attention_temperature_utils.py  # Answer processing and scoring utilities
```

## Installation

### Dependencies

- Python 3.8+
- vLLM
- datasets (Hugging Face)
- pandas
- numpy

### Installation Steps

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <project-directory>
   ```

2. Install dependencies:
   ```bash
   pip install vllm datasets pandas numpy
   ```

3. Download datasets (if local version is needed):
   - Place the MATH-500 dataset in `./datasets/MATH-500/test.jsonl`

## Usage

### Running Inference

Use `inference.py` to perform inference on a specified model and dataset:

```bash
python inference.py \
  --model_name "Qwen/Qwen3-0.6B" \
  --dataset "HuggingFaceH4/MATH-500" \
  --do_sample \
  --num_samples 10 \
  --temperature 0.8 \
  --output_base_dir "results"
```

Parameter descriptions:
- `--model_name`: Model name (supports Hugging Face models)
- `--dataset`: Dataset name
- `--num_samples`: Number of samples per problem
- `--temperature`: Sampling temperature
- `--output_base_dir`: Output directory

### Batch Running Experiments

Use the `scripts/run.sh` script to batch run experiments for multiple models, datasets, and temperatures:

```bash
bash scripts/run.sh
```

The script will automatically run inference for the configured models and temperatures.

### Evaluating Results

Run evaluation scripts to calculate accuracy:

```bash
python eval/majority_vote.py
```

Other evaluation methods are similar; run the corresponding Python scripts.

## Output Format

Inference results are saved in JSONL files, with each record containing:
- `global_index`: Problem index
- `problem`: Problem text
- `boxed_content`: Model's boxed answer output
- `extracted_answer`: Extracted answer
- `target_answer`: Correct answer
- `score`: Score (0 or 1)
- `raw_output`: Raw model output

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For questions, please contact via GitHub Issues.

## 贡献

欢迎贡献！请遵循以下步骤：

1. Fork 本仓库
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

## 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

## 联系

如有问题，请通过 GitHub Issues 联系。