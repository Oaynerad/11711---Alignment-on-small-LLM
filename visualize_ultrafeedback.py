import matplotlib.pyplot as plt
from datasets import load_dataset
import pandas as pd

# 1) 载入数据
ds = load_dataset("trl-lib/ultrafeedback_binarized", split="train")

# 2) 取出分数并计算 gap
scores = {
    "score_chosen": ds["score_chosen"],
    "score_rejected": ds["score_rejected"],
}
df = pd.DataFrame(scores)
df["score_gap"] = df["score_chosen"] - df["score_rejected"]

# 3) 打印统计
print(df["score_gap"].describe())

# 4) 画直方图
plt.figure(figsize=(8, 5))
plt.hist(df["score_gap"], bins=50)
plt.xlabel("score_gap (chosen − rejected)")
plt.ylabel("Frequency")
plt.title("UltraFeedback score_gap distribution")
plt.grid(True)
plt.tight_layout()
plt.show()
