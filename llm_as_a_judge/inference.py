import os
import dotenv
import pandas as pd
import argparse
from llm_as_a_judge.llm_as_a_judge import LLMAsAJudge

dotenv.load_dotenv()


if __name__ == "__main__":
    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--provider", type=str, default="openai")
    parser.add_argument("--dataset_name", type=str, default="wmt-human")
    args = parser.parse_args()

    # Model configuration
    model_config = {
        "mistral": {
            "model": args.model,
            "api_key": os.environ["MISTRAL_API_KEY"]
        },
        "openai": {
            "model": args.model,
            "api_key": os.environ["OPENAI_API_KEY"]
        }
    }

    # LLM-as-a-judge system prompt
    system_prompt = "You are a fair and knowledgeable judge."

    # Read the dataset
    df = pd.read_csv(f"../{args.dataset_name}/data/wmt-human_en_de.csv") \
        if args.dataset_name == "wmt-human" \
            else pd.read_csv(f"../{args.dataset_name}/data/newsroom-human-eval-converted.csv")

    # Run the inference
    llm_as_a_judge = LLMAsAJudge(
        model=model_config[args.provider].get("model"),
        api_key=model_config[args.provider].get("api_key"),
        system_prompt=system_prompt,
        prompt_file_path=f"../{args.dataset_name}/prompts/prompt.txt"
    )
    llm_as_a_judge.run_inference(args.dataset_name, df)

    # Save the results
    df.to_csv(
        f"../{args.dataset_name}/data/llm_as_a_judge_{args.provider}_{args.model}.csv",
        index=False
    )