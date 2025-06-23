from mistralai import Mistral
import openai
import pandas as pd


class LLMAsAJudge:
    """
    LLM-as-a-judge is a class that allows you to use a LLM as a judge.

    Args:
        model: The model to use.
        api_key: The API key to use.
        system_prompt: The system prompt to use.
        prompt_file_path: The path to the prompt file.
    """

    def __init__(self, model: str, api_key: str, system_prompt: str, prompt_file_path: str):
        self.model = model
        self.api_key = api_key
        self.system_prompt = system_prompt
        self.prompt = self.read_prompt(prompt_file_path)

        if "mistral" in model:
            self.provider = "mistral"
            self.client = Mistral(api_key=api_key)

        elif "gpt" in model:
            self.provider = "openai"
            openai.api_key = api_key
        else:
            raise ValueError(f"❌ Unsupported model: {model}.")

    def read_prompt(self, prompt_file_path: str) -> str:
        """Read the prompt from a file.
        """
        with open(prompt_file_path, "r") as f:
            return f.read()

    def run_inference(self, dataset_name: str, df: pd.DataFrame) -> str:
        """Run LLM inference.
        """
        for idx, row in df.iterrows():

            if dataset_name == "wmt-human":
                prompt = prompt.replace("{{ source }}", row["source"])
                prompt = prompt.replace("{{ reference }}", row["reference"])
                prompt = prompt.replace("{{ translation }}", row["translation"])

            elif dataset_name == "wmt-machine":
                prompt = prompt.replace("{{ instance }}", row["instance"])
            else:
                raise ValueError(f"❌ Unsupported dataset: {dataset_name}.")

            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self.prompt}
            ]

            if self.provider == "mistral":
                completion = self.client.chat.complete(
                    model=self.model,
                    messages=messages
                )
                df.at[idx, f"{self.model}_as_a_judge"] = completion.choices[0].message.content

            elif self.provider == "openai":
                completion = openai.chat.completions.create(
                    model=self.model,
                    messages=messages
                )
                df.at[idx, f"{self.model}_as_a_judge"] = completion.choices[0].message.content

            else:
                raise ValueError(f"❌ Unsupported provider: {self.provider}.")

        return df