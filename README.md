# ***`When is an LLM-as-a-judge good enough?`***
---
---
# ***`Abstract`***

*Generative AI models allow us to generate human-like content, such as large language models (LLMs) generate texts. However, such created contend can only be used to automate processes if certain requirements regarding trustworthiness and correctness are fulfilled. In order to relieve people from controlling the created content, a concept known as LLM-as-a-judge is available. In this scenario, another instance of an LLM is prompted to act as a judge, and check a created text for certain quality requirements. However, such a judgement might not always align with a human judgement, which requires benchmarking the LLM-judge as well. Since a full alignment might not be achieved, the question is answered in this work, when an LLM-as-a-judge is good enough to judge a specific task fulfillment to break evaluating evaluations. For this purpose, it is made use of the fact that for evaluating generative content, judgements from different human evaluators can also differ as there might be no universal or unequivocally assessment of the quality, called a diversity of opinion. As long as the deviations of the LLM-judge from human estimations remain within the human diversity of opinion, it is suggested to call an LLM-judge as good enough.*

---

> ***`Paper accessible at the following link:`*** xxxxxxxx.xx

## Getting Started

### 1. Install dependencies

```bash
conda create --name llm-good-enough python=3.11.9
pip install -r requirements.txt
```

### 2. Usage Example

```python
import pandas as pd
from llm_good_enough import LLMGoodEnough

# Load your DataFrame (replace with your own data)
df = pd.read_csv('your_data.csv')
llm_as_a_judge_col = "GPT_as_a_judge"
human_cols = ['human_#1', 'human_#2', 'human_#3']  # replace with your human judge columns
min_score, max_score = 1, 5  # adjust to your rating scale

# Initialize evaluator
LLM_Evaluator = LLMGoodEnough(df=df, human_cols=human_cols, min_score=min_score, max_score=max_score)

# visualize LLM-as-a-judge good enough
LLM_Evaluator.visulize_good_enough(
    llm_col=llm_as_a_judge_col,        # LLM-as-a-judge column name
    y_lim=0.6,                         # y-axis limit -> probability (0-1)
)
```

`See the the folloiwng notebooks for more detailed examples:`

    - LLM-as-a-Judge/LLM-as-a-Judge-good-enough/llm_good_enough/example.ipynb
    - LLM-as-a-Judge/LLM-as-a-Judge-good-enough/wmt-human/src/analysis.ipynb
    - LLM-as-a-Judge/LLM-as-a-Judge-good-enough/newsroom/src/analysis.ipynb