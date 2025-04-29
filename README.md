# 11711---Alignment-on-small-LLM

## Explanation for the code

We use `train_dpo.py` to train `Qwen2.5-0.5B-Instruct` on an EC2 instance. Then, we use `generate_alpaca_eval_with_token.py` to generate AlpacaEval responses with vllm. 
- `create_figure.ipynb`: MT-bench radar figure (The corresponding data is stored in `MT-bench_all_models.jsonl`.)
- All the AlpacaEval results are in this repository. They are named as `alpaca_eval_{method}`, most of them are stored in a folder (containing the annotation and the leaderboard), for some of them we only saved the leaderboard (the csv file).
- `test_davinci_003_output` is the baseline for AlpacaEval 1.0

## Contributions
Daren Yao | darenyao@andrew.cmu.edu
- Designed the testing pipeline for alignment methods (TRL DPO trainer -> Alpaca Eval & MT-bench).
- Proposed APO-hinge and APO-hinge-softmax
- Training: APO-hinge, DPO, DPO-hinge
- Report: Analyze results, Discussion, visualization and consolidate report content.
  
Jinsong Yuan | jinsongy@andrew.cmu.edu
- Proposed margin-sigmoid method
- Training: AOT, IPO, margin-sigmoid
- Report: Analyze training results, Discussion and Conclusions
  
Ruike Chen | ruikec@andrew.cmu.edu
- Training: APO, EXO, APO-hinge-softplus
- Report: Introduction Related work and AlpacaEval analysis.

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
