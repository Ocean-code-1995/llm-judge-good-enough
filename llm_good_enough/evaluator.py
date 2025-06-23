import pandas as pd
import numpy as np
import random
from itertools import combinations
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")


class LLMGoodEnough:
    """
    Is your selected LLM-as-a-judge good enough?

    This class evaluates whether an LLM's performance is "good enough" and hence suitbale for automated evaluation tasks of other LLM generated outputs. It achieves this by creating two arrays of absolute differences between inter-human judgements as well as LLM-human judgements. Finally, LLM-human judgements are then compared to the inter-human judgements using a Mann-Whitney U test in order to determine if the selected candidate LLM is good enough.

    Corresponding paper: https://arxiv.org/abs/--->>>ToBeAnnounced<<<---

    Example:
    ```python
    from llm_good_enough import LLMGoodEnough
    LLM_Evaluator = LLMGoodEnough(df=df, human_cols=human_cols)
    ```
    """

    def __init__(self, df: pd.DataFrame, human_cols: list[str], min_score: int, max_score: int) -> None:
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
        min_score : int
            Minimum score for the random judge.
        max_score : int
            Maximum score for the random judge.
        """
        self.df = df
        self.human_cols = human_cols
        self.min_score = min_score
        self.max_score = max_score
        self.df['RANDOM_as_a_judge'] = self.init_random_judge(min_score=self.min_score, max_score=self.max_score)

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
            print(f"❌ Data contains NaN values in {llm_col} column.")
            # drop rows with nan in llm_col
            print(f"✅Dropping {self.df[self.df[llm_col].isna()].shape[0]} rows with nan in {llm_col} column.")
            self.df = self.df.dropna(subset=[llm_col])
            #raise ValueError("❌ Data contains NaN values in required columns.")

        # Convert to NumPy arrays
        human_ratings = self.df[self.human_cols].to_numpy()
        llm_ratings = self.df[llm_col].to_numpy().reshape(-1, 1)  # column vector for broadcasting

        # Compute absolute differences: each human col vs. LLM
        differences = np.abs(human_ratings - llm_ratings)

        return differences.flatten()


    @staticmethod
    def run_mannwhitneyu_test(
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


    def init_random_judge(self, min_score: int, max_score: int, seed: int = 42) -> np.ndarray:
        """
        Initialize a random judge as a comparison to the LLM-human judgements. The random judge
        is initialized with a random integer between the minimum and maximum value.

        Parameters
        ----------
        min_score : int
            Minimum value for the random judge.
        max_score : int
            Maximum value for the random judge.

        Returns
        -------
        np.ndarray
            Array of random integers between the minimum and maximum value.
        """
        np.random.seed(seed)
        return np.random.randint(min_score, max_score + 1, size=len(self.df))


    def visulize_good_enough(self, llm_col: str, y_lim: float, save_path: str = None) -> None:
        """
        Visualize the LLM-as-a-judge performance compared to human judges and a random baseline.

        Parameters
        ----------
        llm_col : str
            Column name for the LLM judge's ratings.
        min_score : int, optional
            Minimum possible score in the rating scale. If None, will be inferred from data.
        max_score : int, optional
            Maximum possible score in the rating scale. If None, will be inferred from data.
        y_lim : float
            Upper limit of the y-axis.
        save_path : str, optional
            Path to save the figure. If None, the figure will not be saved.
        """
        # 1) Compute disagreement distributions
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 1.1) human-human
        human_human_disagreements = self.compute_human_disagreements()

        # 1.2) llm-human
        llm_human_disagreements = self.compute_llm_human_disagreements(llm_col)

        # random judge
        random_judge = self.init_random_judge(min_score=self.min_score, max_score=self.max_score)
        self.df['RANDOM_as_a_judge'] = random_judge
        random_judge_disagreements = self.compute_llm_human_disagreements('RANDOM_as_a_judge')

        model_vs_human_distributions = {
            'llm-human': llm_human_disagreements,
            'random-human': random_judge_disagreements
        }

        # 3) Plot
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        fig.suptitle(f"LLM-as-a-judge good enough?", fontsize=22, fontweight='bold')
        axes = axes.flatten()

        # Dynamic bins based on score range
        max_possible_disagreement = self.max_score - self.min_score
        bins = np.arange(0, max_possible_disagreement + 2)  # +2 to include the max disagreement value
        bar_width = 0.35
        categories = np.arange(len(bins) - 1)

        for ax, (name, data) in zip(axes, model_vs_human_distributions.items()):
            human_human_mean, human_human_std = round(np.mean(human_human_disagreements), 2), round(np.std(human_human_disagreements), 2)
            llm_human_mean, llm_human_std = round(np.mean(data), 2), round(np.std(data), 2)

            # Calculate p-value for this comparison
            p_val = self.run_mannwhitneyu_test(data, human_human_disagreements)

            # Plot human-human
            ax.hist(
                human_human_disagreements, bins=bins, density=True, alpha=0.6, color='royalblue',
                label=f"Humans' diversity of opinion\nμ = {human_human_mean}, σ = {human_human_std}",
                width=bar_width, edgecolor='black', hatch="///"
            )

            # Plot llm-human
            ax.hist(
                data, bins=bins - 0.35, density=True, alpha=0.5, color='red',
                label=f'Human-Model deviation\nμ = {llm_human_mean}, σ = {llm_human_std}',
                width=bar_width, edgecolor='black', hatch=""
            )

            # Add p-value to legend
            handles, labels = ax.get_legend_handles_labels()
            handles.append(plt.Line2D([], [], color='none'))

            legend = ax.legend(
                handles=handles, labels=labels, loc='upper center', title=f"p-value = {p_val:.4f}",
                edgecolor='black', facecolor='white', framealpha=1,
                fontsize=14.5
                )
            legend.get_title().set_fontweight('bold')
            legend.get_title().set_fontsize(13.5)

            ax.set_title(name, fontweight='bold', fontsize=20)
            ax.set_xlabel('Degree of Disagreement', fontsize=17)
            ax.set_ylabel('Probability', fontsize=17)
            ax.set_xticks(categories)
            ax.set_yticks(np.arange(0, 0.6, 0.1))
            ax.tick_params(axis='both', which='major', labelsize=13)
            ax.set_xlim(min(categories) - bar_width - 0.1, max(categories) + bar_width + 0.1)
            ax.set_ylim(0, y_lim)

        plt.tight_layout()
        plt.show();

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")

    def plot_model_vs_human_grid(
        self,
        model_disagreement_dict: dict,
        human_human_disagreements: np.ndarray = None,
        bins: np.ndarray = None,
        bar_width: float = 0.35,
        y_lim: float = 0.6,
        save_path: str = None
    ):
        """
        Plots a grid of model-vs-human disagreement histograms.

        Parameters
        ----------
        model_disagreement_dict : dict
            Dictionary where keys are model names and values are disagreement arrays.
        human_human_disagreements : np.ndarray, optional
            Array of human-human disagreements. If None, will be computed.
        bins : np.ndarray, optional
            Bins for the histogram. If None, will be inferred from data.
        bar_width : float, optional
            Width of the bars in the histogram.
        y_lim : float, optional
            Upper limit of the y-axis.
        save_path : str, optional
            Path to save the figure. If None, the figure will not be saved.
        """
        if human_human_disagreements is None:
            human_human_disagreements = self.compute_human_disagreements()
        if bins is None:
            min_score = int(self.df[self.human_cols].min().min())
            max_score = int(self.df[self.human_cols].max().max())
            max_possible_disagreement = max_score - min_score
            bins = np.arange(0, max_possible_disagreement + 2)

        n_models = len(model_disagreement_dict)
        n_cols = 2
        n_rows = (n_models + 1) // 2

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5 * n_rows))
        axes = axes.flatten()

        categories = np.arange(len(bins) - 1)

        for ax, (name, data) in zip(axes, model_disagreement_dict.items()):
            mean_model = round(np.mean(data), 2)
            std_model = round(np.std(data), 2)
            mean_human = round(np.mean(human_human_disagreements), 2)
            std_human = round(np.std(human_human_disagreements), 2)
            p_val = self.run_mannwhitneyu_test(data, human_human_disagreements)

            # Plot human-human
            ax.hist(human_human_disagreements, bins=bins, density=True, alpha=0.6, color='royalblue',
                    label=f"Humans' diversity of opinion\nμ = {mean_human}, σ = {std_human}",
                    width=bar_width, edgecolor='black', hatch="///")

            # Plot model-human
            ax.hist(data, bins=bins - 0.35, density=True, alpha=0.5, color='red',
                    label=f'Human-Model deviation\nμ = {mean_model}, σ = {std_model}',
                    width=bar_width, edgecolor='black', hatch="")

            # Add p-value to legend
            handles, labels = ax.get_legend_handles_labels()
            handles.append(plt.Line2D([], [], color='none'))
            legend = ax.legend(
                handles=handles, labels=labels, loc='upper center', title=f"p-value = {p_val:.4f}",
                edgecolor='black', facecolor='white', framealpha=1,
                fontsize=13.5
            )
            legend.get_title().set_fontweight('bold')
            legend.get_title().set_fontsize(13.5)

            ax.set_title(name, fontweight='bold', fontsize=20)
            ax.set_xlabel('Degree of Disagreement', fontsize=13.5)
            ax.set_ylabel('Probability', fontsize=13.5)
            ax.set_xticks(categories)
            ax.set_xlim(min(categories) - bar_width - 0.1, max(categories) + bar_width + 0.1)
            ax.set_ylim(0, y_lim)

        # Hide any unused subplots
        for i in range(len(model_disagreement_dict), len(axes)):
            fig.delaxes(axes[i])

        plt.tight_layout()
        plt.show()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")