import os
import dotenv
import pandas as pd
import argparse
from llm_as_a_judge import LLMAsAJudge
import logging

dotenv.load_dotenv()

logging.basicConfig(level=logging.INFO)


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
    # set model, provider, dataset_name
    # ------------------------------------------------------------
    PROVIDER     = args.provider
    MODEL        = model_config[PROVIDER].get("model")
    API_KEY      = model_config[PROVIDER].get("api_key")
    DATASET_NAME = args.dataset_name.lower()
    SYSTEM_PROMPT = "You are a fair and knowledgeable judge."
    # ------------------------------------------------------------

    # Read the dataset
    df = pd.read_csv(f"../{DATASET_NAME}/data/wmt-human_en_de.csv") \
        if DATASET_NAME == "wmt-human" \
            else pd.read_csv(f"../{DATASET_NAME}/data/newsroom-human-eval-converted.csv")

    # for testing just take the first 10 rows
    df = df.head(3)


    logging.info(f"✅ Loaded {len(df)} rows from {DATASET_NAME}")


    # Run the inference
    logging.info(f"🚀 Running inference for {DATASET_NAME} with {PROVIDER} {MODEL}.")
    llm_as_a_judge = LLMAsAJudge(
        model=MODEL,
        api_key=API_KEY,
        system_prompt=SYSTEM_PROMPT,
        prompt_file_path=f"../{DATASET_NAME}/prompts/prompt.txt"
    )
    df = llm_as_a_judge.run_inference(dataset_name=DATASET_NAME, df=df)
    logging.info(f"✅ Inference completed.")

    # Save the results
    df.to_csv(
        path_or_buf=f"../{DATASET_NAME}/data/llm_as_a_judge_{PROVIDER}_{MODEL}.csv",
        index=False
    )
    logging.info(f"💾 Saved results to {DATASET_NAME}/data/llm_as_a_judge_{PROVIDER}_{MODEL}.csv")