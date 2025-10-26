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

    This class evaluates whether an LLM's performance is "good enough" 
    and hence suitbale for automated evaluation tasks of other LLM generated outputs. 
    It achieves this by creating two arrays of absolute differences between inter-human 
    judgements as well as LLM-human judgements. Finally, LLM-human judgements are then compared 
    to the inter-human judgements using a Mann-Whitney U test in order to determine 
    if the selected candidate LLM is good enough.

    Corresponding paper: https://arxiv.org/abs/--->>>ToBeAnnounced<<<---

    Example:
    ```python
    from llm_good_enough import LLMGoodEnough
    LLM_Evaluator = LLMGoodEnough(df=df, human_cols=human_cols)
    ```
    """

    def __init__(
            self, df: pd.DataFrame,
            human_cols: list[str],
            min_score: int,
            max_score: int
        ) -> None:
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
        self.df = df.copy()
        self.human_cols = human_cols
        self.min_score = min_score
        self.max_score = max_score

        # --- Initialization pipeline ---
        self._validate_human_columns()
        self._filter_minimum_raters(min_raters=2)
        self._add_random_judge()
    
    def _validate_human_columns(self) -> None:
        """
        Validate that all specified human rating columns exist in the input DataFrame
        and that at least two are provided.

        Raises
        ------
        ValueError
            If any specified column name is missing from the DataFrame.
        ValueError
            If fewer than two human columns are provided.

        Notes
        -----
        This check ensures that the dataset has enough valid human annotators
        to compute inter-human disagreement distributions.
        """
        missing_cols = [c for c in self.human_cols if c not in self.df.columns]
        if missing_cols:
            raise ValueError(f"❌ Human columns not found in DataFrame: {missing_cols}")

        if len(self.human_cols) < 2:
            raise ValueError("❌ At least two human columns are required for disagreement analysis.")

        print(f"✅ Found {len(self.human_cols)} valid human annotator columns.")


    def _filter_minimum_raters(self, min_raters: int = 2) -> None:
        """
        Filter the dataset to include only cases (rows) that have at least a
        minimum number of non-missing human ratings.

        Parameters
        ----------
        min_raters : int, default=2
            Minimum number of human annotators required for a case to be retained.

        Raises
        ------
        ValueError
            If no rows remain after filtering for the specified number of annotators.

        Notes
        -----
        This method ensures that each remaining case can produce at least one
        valid inter-human pairwise difference. Cases with fewer than `min_raters`
        annotators are discarded, as they cannot contribute to the comparison.
        """
        # Count non-null human ratings per row
        self.df = self.df.copy()  # ensures we don't modify a view
        self.df.loc[:, "n_raters"] = self.df[self.human_cols].notna().sum(axis=1)
        n_before = len(self.df)

        # Keep only rows with enough human raters
        self.df = self.df[self.df["n_raters"] >= min_raters].copy()
        n_after = len(self.df)

        if n_after == 0:
            raise ValueError(f"❌ No rows have ≥{min_raters} human annotators. Cannot proceed.")

        print(f"✅ Keeping {n_after}/{n_before} rows with ≥{min_raters} human annotators.")


    def _add_random_judge(self) -> None:
        """
        Add a baseline random judge column to the dataset.

        The random judge produces uniformly random integer scores in the
        same rating range as the human and LLM annotators. This provides
        a comparison baseline for the LLM-as-a-judge analysis.

        Notes
        -----
        This method calls `init_random_judge()` internally and adds the resulting
        column ('RANDOM_as_a_judge') to `self.df`.
        """
        self.df["RANDOM_as_a_judge"] = self.init_random_judge(
            min_score=self.min_score,
            max_score=self.max_score
        )
        print("✅ Added random judge baseline column: 'RANDOM_as_a_judge'")


    # --- Core disagreement computations ---
    def compute_human_disagreements(self) -> np.ndarray:
        """
        Vectorized computation of all available pairwise absolute differences 
        between human raters per case.

        Returns
        -------
        np.ndarray
            Flattened array of all pairwise absolute disagreements.
        """

        diffs = []
        for _, row in self.df[self.human_cols].iterrows():
            # extract non-nan values
            vals = row.dropna().to_numpy()
            # only compute if at least 2 values (raters) are present
            if len(vals) >= 2:
                # upper triangle pairwise differences
                diffs.extend(
                    np.abs(vals[:, None] - vals[None, :])[np.triu_indices(len(vals), k=1)]
                )
        if not diffs:
            raise ValueError(
                "❌ No valid human-human pairs found (check your input data)."
            )
        return np.array(diffs)


    def compute_llm_human_disagreements(self, llm_col: str) -> np.ndarray:
        """
        Compute absolute differences between LLM and all available human judges per case.

        Parameters
        ----------
        llm_col : str
            Column name for the LLM judge.

        Returns
        -------
        np.ndarray
            Flattened array of LLM-human absolute disagreements.
        """
        if llm_col not in self.df.columns:
            raise ValueError(f"❌ LLM column not found in DataFrame: {llm_col}")

        diffs = []
        # iterate over each row
        for _, row in self.df[[llm_col] + self.human_cols].iterrows():
            # extract LLM value
            llm_val = row[llm_col]
            # only compute if LLM value is not nan
            if pd.notna(llm_val):
                vals = row[self.human_cols].dropna().to_numpy()
                diffs.extend(
                    np.abs(vals - llm_val)
                )
        if not diffs:
            raise ValueError(
                "❌ No valid LLM-human pairs found (check for missing ratings)."
            )
        return np.array(diffs)


    # --- Utility methods ---
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

    # --- Summary stats ---
    def summarize(self, llm_col: str) -> None:
        """
        Summarize the disagreement statistics.
        
        Parameters
        ----------
        llm_col : str
            Column name for the LLM judge's ratings.
        """
        human_diffs = self.compute_human_disagreements()
        llm_diffs = self.compute_llm_human_disagreements(llm_col)
        p_val = self.run_mannwhitneyu_test(llm_diffs, human_diffs)
        print(f"Human–Human mean: {np.mean(human_diffs):.2f}")
        print(f"LLM–Human mean: {np.mean(llm_diffs):.2f}")
        print(f"Mann–Whitney p-value: {p_val:.4f}")



    # --- Visualization methods ---
    def visualize_good_enough(self, llm_col: str, y_lim: float, save_path: str = None) -> None:
        """
        Visualize the LLM-as-a-judge performance compared to human judges and a random baseline.

        Parameters
        ----------
        llm_col : str
            Column name for the LLM judge's ratings.
        y_lim : float
            Upper limit of the y-axis.
        save_path : str, optional
            Path to save the figure. If None, the figure will not be saved.
        """
        print("📊 Computing disagreement distributions...")

        # 1) Compute disagreement distributions
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        human_human_disagreements = self.compute_human_disagreements()
        llm_human_disagreements = self.compute_llm_human_disagreements(llm_col)
        random_human_disagreements = self.compute_llm_human_disagreements("RANDOM_as_a_judge")

        model_vs_human_distributions = {
            "llm-human": llm_human_disagreements,
            "random-human": random_human_disagreements
        }

        # 2) Plot setup
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        fig.suptitle("LLM-as-a-judge good enough?", fontsize=22, fontweight="bold")
        axes = axes.flatten()

        max_possible_disagreement = self.max_score - self.min_score
        bins = np.arange(0, max_possible_disagreement + 2)
        bar_width = 0.35
        categories = np.arange(len(bins) - 1)

        # 3) Plot histograms
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        for ax, (name, data) in zip(axes, model_vs_human_distributions.items()):
            # Compute stats
            human_mean, human_std = round(np.mean(human_human_disagreements), 2), round(np.std(human_human_disagreements), 2)
            model_mean, model_std = round(np.mean(data), 2), round(np.std(data), 2)
            p_val = self.run_mannwhitneyu_test(data, human_human_disagreements)

            # Plot human-human disagreements
            ax.hist(
                human_human_disagreements, bins=bins, density=True, alpha=0.6, color="royalblue",
                label=f"Humans' diversity of opinion\nμ = {human_mean}, σ = {human_std}",
                width=bar_width, edgecolor="black", hatch="///"
            )

            # Plot model-human disagreements
            ax.hist(
                data, bins=bins - 0.35, density=True, alpha=0.5, color="red",
                label=f"Human-Model deviation\nμ = {model_mean}, σ = {model_std}",
                width=bar_width, edgecolor="black", hatch=""
            )

            # Add p-value to legend
            #handles, labels = ax.get_legend_handles_labels()
            #handles.append(plt.Line2D([], [], color="none"))
            legend = ax.legend(
                #handles=handles, labels=labels, 
                loc="upper center", title=f"p-value = {p_val:.4f}",
                edgecolor="black", facecolor="white", framealpha=1, fontsize=14.5
            )
            legend.get_title().set_fontweight("bold")
            legend.get_title().set_fontsize(13.5)

            # Axis formatting (identical to original style)
            ax.set_title(name, fontweight="bold", fontsize=20)
            ax.set_xlabel("Degree of Disagreement", fontsize=17)
            ax.set_ylabel("Probability", fontsize=17)
            ax.set_xticks(categories)
            ax.set_yticks(np.arange(0, 0.6, 0.1))
            ax.tick_params(axis="both", which="major", labelsize=13)
            ax.set_xlim(min(categories) - bar_width - 0.1, max(categories) + bar_width + 0.1)
            ax.set_ylim(0, y_lim)

        plt.tight_layout()

        # 4) Save figure if requested
        if save_path:
            if "." in save_path:
                ext = save_path.split(".")[-1].lower()
            else:
                ext = "pdf"
                save_path += ".pdf"
            plt.savefig(save_path, dpi=300, bbox_inches="tight", format=ext)
            print(f"✅ Figure saved as {ext.upper()} → {save_path}")

        plt.show()



    def plot_model_vs_human_grid(
        self,
        model_disagreement_dict: dict,
        human_human_disagreements: np.ndarray = None,
        bins: np.ndarray = None,
        bar_width: float = 0.35,
        y_lim: float = 0.6,
        save_path: str = None
    ) -> None:
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

        if save_path:
            # infer format automatically if not provided
            if "." in save_path:
                ext = save_path.split(".")[-1].lower()
            else:
                ext = "pdf"  # default to PDF if no extension given
                save_path += ".pdf"

            plt.savefig(save_path, dpi=300, bbox_inches='tight', format=ext)
            print(f"✅ Figure saved as {ext.upper()} → {save_path}")

        plt.show()