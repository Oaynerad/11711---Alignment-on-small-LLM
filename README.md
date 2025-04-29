# 11711---Alignment-on-small-LLM

## Explanation for the code

We use `train_dpo.py` to train `Qwen2.5-0.5B-Instruct` on an EC2 instance. Then, we use `generate_alpaca_eval_with_token.py` to generate AlpacaEval responses with vllm. 
- `create_figure.ipynb`: MT-bench radar figure
- 

## Partial Content

For more information, please refer to `report.pdf`
Small language models (LLMs) often face difficulties in aligning output to human preferences, particularly when operating under severe performance gaps.
In this work, we propose two lightweight DPO-based variants---Adaptive Margin-Sigmoid Loss and APO-Hinge---to better address underperformance scenarios by introducing margin-based objectives and selective update mechanisms.

Our APO-Hinge method, which combines hinge-induced hard-example mining with the chosen-focused optimization of APO-Zero, achieves strong results.
In AlpacaEval, APO-Hinge improves the win rate by +2.0 points and the length-controlled win rate by +1.4 points compared to the APO baseline.
In MT-Bench, our methods maintain competitive performance in diverse categories, particularly excelling in STEM and Humanities tasks.

These results demonstrate that simple modifications to preference-based objectives can significantly enhance small LLM alignment under resource constraints, offering a practical path toward more efficient deployment.



![fig](https://github.com/user-attachments/assets/9b9e82cb-a3d1-4b68-b371-cba2269fb6f8)
![mtbench1](https://github.com/user-attachments/assets/4325e633-b678-4fd8-9150-2bb46122b60a)
![output (2)](https://github.com/user-attachments/assets/c75c4daa-e7df-4d62-9af1-7c24b6dc82b7)
