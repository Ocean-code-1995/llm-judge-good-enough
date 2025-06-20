import pandas as pd
import numpy as np
from itertools import combinations
from scipy.stats import mannwhitneyu


class LLMGoodEnough:
    """
    Is your selected LLM-as-a-judge good enough?

    This class evaluates whether an LLM's performance is "good enough" by comparing
    LLM-human disagreements to human-human disagreements using statistical testing.

    Corresponding paper: https://arxiv.org/abs/--->>>ToBeAnnounced<<<---
    """

    def __init__(self, df: pd.DataFrame, human_cols: list[str]) -> None:
        """
        Initialize the LLM Good Enough evaluator instance.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing human and LLM ratings. Each column should represent
            ratings from a different judge (human or LLM).
        human_cols : list of str
            Column names for human raters. These are the "ground truth" judges that
            LLM performance will be compared against.
        """
        self.df = df
        self.human_cols = human_cols

        # Validate human columns exist
        missing_cols = [col for col in human_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"❌ Human columns not found in DataFrame: {missing_cols}"
            )

        if len(human_cols) < 2:
            raise ValueError("❌ At least two human columns are required.")

        # Check for NaN values in human columns
        if df[human_cols].isna().any().any():
            raise ValueError("❌ Data contains NaN values in human columns.")

    def compute_human_disagreements(self) -> np.ndarray:
        """
        Vectorized computation of all pairwise absolute differences between human raters per case.

        Returns
        -------
        np.ndarray
            Flattened array of all pairwise absolute disagreements.
        """

        # Select only human columns and convert to numpy
        data = self.df[self.human_cols].to_numpy()

        # Compute pairwise differences per row for all combinations
        comb_indices = list(combinations(range(data.shape[1]), 2))

        diffs = np.abs([data[:, i] - data[:, j] for i, j in comb_indices])
        return diffs.flatten()


    def compute_llm_human_disagreements(self, llm_col: str) -> np.ndarray:
        """
        Vectorized computation of absolute differences between LLM and each human judge per case.

        Parameters
        ----------
        llm_col : str
            Column name for the LLM judge's ratings.

        Returns
        -------
        np.ndarray
            Array of absolute differences (LLM vs. each human) across all cases.
        """
        if llm_col not in self.df.columns:
            raise ValueError(f"❌ LLM column not found in DataFrame: {llm_col}")

        # Check for NaN values in llm column
        if self.df[llm_col].isna().any().any():
            raise ValueError("❌ Data contains NaN values in required columns.")

        # Convert to NumPy arrays
        human_ratings = self.df[self.human_cols].to_numpy()
        llm_ratings = self.df[llm_col].to_numpy().reshape(-1, 1)  # column vector for broadcasting

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
        human_human_disagreements : np.ndarray
            Array of absolute differences between human judges.

        Returns
        -------
        float
            P-value from Mann-Whitney U test. Lower values indicate LLM disagreements
            are significantly greater than human disagreements.
        """
        # compute mann-whitney u test and round to 4 decimal places
        return mannwhitneyu(
            x=llm_human_disagreements,
            y=human_human_disagreements,
            alternative='greater'
        ).pvalue

    def visulize_good_enough(self):
        """
        """
        pass