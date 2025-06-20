import pandas as pd
import numpy as np
from itertools import combinations
from scipy.stats import mannwhitneyu

class LLMGoodEnough:
    def __init__(self, df: pd.DataFrame) -> None:
        """
        """
        self.df = df

    def compute_human_disagreements(self, human_cols: list[str]) -> np.ndarray:
        """
        Vectorized computation of all pairwise absolute differences between human raters per case.

        Parameters
        ----------
        human_cols : list of str
            Column names for human raters.

        Returns
        -------
        np.ndarray
            Flattened array of all pairwise absolute disagreements.
        """
        if len(human_cols) < 2:
            raise ValueError("❌ At least two human columns are required.")

        # Select only human columns and convert to numpy
        data = self.df[human_cols].to_numpy()

        if np.isnan(data).any():
            raise ValueError("❌ Data contains NaN values.")

        # Compute pairwise differences per row for all combinations
        comb_indices = list(combinations(range(data.shape[1]), 2))

        diffs = np.abs([data[:, i] - data[:, j] for i, j in comb_indices])
        return diffs.flatten()


    def compute_llm_human_disagreements(self, human_cols: list[str], llm_col: str) -> np.ndarray:
        """
        Vectorized computation of absolute differences between LLM and each human judge per case.

        Parameters
        ----------
        human_cols : list of str
            Column names for human judges.
        llm_col : str
            Column name for the LLM judge's ratings.

        Returns
        -------
        np.ndarray
            Array of absolute differences (LLM vs. each human) across all cases.
        """
        if len(human_cols) < 2:
            raise ValueError("❌ At least two human columns are required.")

        # if nan in df raise error
        if self.df.isna().any().any():
            raise ValueError("❌ Data contains NaN values.")

        # Drop rows with any missing values in required columns
        required_cols = human_cols + [llm_col]
        filtered_df = self.df.dropna(subset=required_cols)

        # Convert to NumPy arrays
        human_ratings = filtered_df[human_cols].to_numpy()
        llm_ratings = filtered_df[llm_col].to_numpy().reshape(-1, 1)  # column vector for broadcasting

        # Compute absolute differences: each human col vs. LLM
        differences = np.abs(human_ratings - llm_ratings)

        return differences.flatten()


    @staticmethod
    def mannwhitney_test(
        llm_human_disagreements: np.ndarray,
        human_human_disagreements: np.ndarray
        ) -> float:
        """
        Compute Mann-Whitney U test for LLM vs. human disagreement.

        Parameters
        ----------
        llm_human_disagreements : np.ndarray
            Array of absolute differences between LLM and each human judge.
        """
        # compute mann-whitney u test and round to 4 decimal places
        return mannwhitneyu(
            llm_human_disagreements, human_human_disagreements, alternative='greater'
        ).pvalue

    def visulize_good_enough(self):
        """
        """
        pass