from turtle import color
import pandas as pd
import numpy as np
import random
from itertools import combinations
from typing import Optional
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

    # ~~~~~~~~~~~~~~~ INITIALIZATION & INTERNAL SETUP ~~~~~~~~~~~~~~~

    def __init__(
            self, df: pd.DataFrame,
            human_cols: list[str],
            llm_cols: list[str],
            min_score: int,
            max_score: int,
            verbosity: int = 1,
            seed: Optional[int] = None,
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
        llm_cols : list of str
            Column names for LLM raters. These are the LLM judges that
            will be compared to the human judges.
        min_score : int
            Minimum score for the random judge.
        max_score : int
            Maximum score for the random judge.
        verbosity : bool, default=True
            If True, print progress and status messages during initialization.
        seed : int, optional
            Random seed for reproducibility. If None, a random seed will be generated.
            Note, that across executions the seeds will differ if None is provided.
        """
        self.df = df.copy()
        self.human_cols = human_cols
        self.llm_cols = llm_cols
        self.min_score = min_score
        self.max_score = max_score
        self.verbosity = verbosity

        # Initialize and apply seed
        self.seed = self._init_seed(seed)
        self._set_global_seed(self.seed)
        if self.verbosity > 0:
            print(f"🎲 Random seed: {self.seed}")

        # --- Initialization pipeline ---
        self._validate_human_columns()
        self._filter_minimum_raters(min_raters=2)
        self._add_random_judge()


    def _init_seed(self, seed: int | None) -> int:
        """
        Initialize or validate the random seed.
        Generates a new valid seed if none is provided.
        Note that across executions the seeds will differ if None is provided.
        """
        if seed is None:
            # Generate a random valid 32-bit integer seed
            return int(np.random.SeedSequence().generate_state(1)[0])
        if not (0 <= seed <= 2**32 - 1):
            raise ValueError("❌ Seed must be between 0 and 2**32 - 1")
        return int(seed)


    def _set_global_seed(self, seed: int) -> None:
        """
        Apply the seed globally for deterministic behavior.
        """
        np.random.seed(seed)
        random.seed(seed)


    def reseed(self, new_seed: int | None = None) -> None:
        """
        Reseed the RNGs mid-session. If no seed provided, generate a new one.
        """
        self.seed = self._init_seed(new_seed)
        self._set_global_seed(self.seed)
        if self.verbosity > 0:
            print(f"🔁 RNGs reseeded with: {self.seed}")

    
    def _reseed_and_refresh(self) -> None:
        """
        Reseed the RNGs and refresh the random judge column in-place.
        This is used for repeated randomization analyses without re-instantiating the class.
        """
        self.reseed(None)
        self.df["RANDOM_as_a_judge"] = self.init_random_judge(
            min_score=self.min_score,
            max_score=self.max_score
        )
    
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
        rng = np.random.default_rng(self.seed)
        return rng.integers(
            min_score, max_score + 1, size=len(self.df)
        )

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
        if self.verbosity > 0:  
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
        if self.verbosity > 0:
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
        if self.verbosity > 0:
            print("✅ Added random judge baseline column: 'RANDOM_as_a_judge'")


    # ~~~~~~~~~~~~~~~ CORE COMPUTATION METHODS (PUBLIC) ~~~~~~~~~~~~~~~
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


    def compute_all_llm_disagreements(self) -> dict:
        """
        Compute all LLM-human disagreements for all LLM columns.

        Returns
        -------
        dict
            Dictionary where keys are LLM column names and values are arrays of LLM-human disagreements.
        """
        llm_disagreements = {
            llm_col: self.compute_llm_human_disagreements(llm_col)
            for llm_col in self.llm_cols
            if llm_col in self.df.columns
        }

        if self.verbosity > 0:
            print(f"✅ Computed LLM-human disagreements for {len(llm_disagreements)} LLM columns.")
        if self.verbosity > 1:
            print(f"LLM-human disagreements computed for: {list(llm_disagreements.keys())}")

        return llm_disagreements


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

    # ~~~~~~~~~~~~~~~ INTERNAL HELPER METHODS (PRIVATE) ~~~~~~~~~~~~~~~
    def _save_figure(self, save_path: str, check_verbosity: bool = False) -> None:
        """
        Helper method to save a matplotlib figure with automatic format detection.
        
        Parameters
        ----------
        save_path : str
            Path to save the figure. Format is inferred from extension, or defaults to PDF.
        check_verbosity : bool, default=False
            If True, only print success message when verbosity > 0.
        """
        if not save_path:
            return
            
        # Infer format automatically if not provided
        if "." in save_path:
            ext = save_path.split(".")[-1].lower()
        else:
            ext = "pdf"  # default to PDF if no extension given
            save_path += ".pdf"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight', format=ext)
        
        if not check_verbosity or self.verbosity > 0:
            print(f"✅ Figure saved as {ext.upper()} → {save_path}")


    # ~~~~~~~~~~~~~~~ INTERNAL PLOTTING ENGINE (private) ~~~~~~~~~~~~~~~
    def _plot_disagreement_grid(
        self,
        model_disagreement_dict: dict,
        human_human_disagreements: np.ndarray = None,
        bins: np.ndarray = None,
        bar_width: float = 0.35,
        y_lim: float = 0.6,
        save_path: Optional[str] = None
    ) -> None:
        """
        INTERNAL PLOTTING ENGINE.
        Handles rendering of model-vs-human disagreement histograms.

        This method should not be called directly by users.

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
            ax.hist(
                human_human_disagreements, bins=bins, density=True, alpha=0.6, color='royalblue',
                label=f"Humans' diversity of opinion\nμ = {mean_human}, σ = {std_human}",
                width=bar_width, edgecolor='black', hatch="///"
            )

            # Plot model-human
            ax.hist(
                data, bins=bins - 0.35, density=True, alpha=0.5, color='red',
                label=f'Human-Model deviation\nμ = {mean_model}, σ = {std_model}',
                width=bar_width, edgecolor='black', hatch=""
            )

            # Add p-value to legend
            #handles, labels = ax.get_legend_handles_labels()
            #handles.append(plt.Line2D([], [], color='none'))
            legend = ax.legend(
                #handles=handles, labels=labels, 
                loc='upper center', title=f"p-value = {p_val:.4f}",
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

        self._save_figure(save_path)
        plt.show()


    # ~~~~~~~~~~~~~~~ VISUALIZATION METHODS (PUBLIC API) ~~~~~~~~~~~~~~~

    def plot_judges_grid(
        self,
        y_lim: float = 0.6,
        save_path: str = None,
        bar_width: float = 0.35,
    ) -> None:
        """
        Public API method.
        Plots a grid comparing all LLM judges + random judge against humans.

        Automatically:
        - computes human-human disagreements
        - computes each LLM's disagreements
        - adds Random Judge baseline
        - delegates rendering to the internal grid plotter
        """

        human_disagreements = self.compute_human_disagreements()

        model_disagreements = self.compute_all_llm_disagreements()
        model_disagreements["Random Judge"] = self.compute_llm_human_disagreements(
            "RANDOM_as_a_judge"
        )

        return self._plot_disagreement_grid(
            model_disagreement_dict=model_disagreements,
            human_human_disagreements=human_disagreements,
            y_lim=y_lim,
            bar_width=bar_width,
            save_path=save_path,
        )


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
        self._save_figure(save_path)
        plt.show()


    # ~~~~~~~~~ MONTE CARLO SIMULATION ~~~~~~~~~~
    def _monte_carlo_random_judges(
        self,
        iterations: int,
        min_score: int,
        max_score: int,
        human_dis: np.ndarray,
    ) -> pd.DataFrame:
        """
        INTERNAL:
        Runs Monte-Carlo simulation of random judges and returns a DataFrame with
        columns:
            - delta_mean
            - p_value
        """
        rng = np.random.default_rng(self.seed)
        results = np.empty((iterations, 2))

        for i in range(iterations):
            random_ratings = rng.integers(min_score, max_score + 1, size=len(human_dis))
            random_dis = np.abs(random_ratings - human_dis)

            delta = np.mean(random_dis) - np.mean(human_dis)
            p_val = mannwhitneyu(random_dis, human_dis, alternative="greater").pvalue

            results[i] = (delta, p_val)

        return pd.DataFrame(results, columns=["delta_mean", "p_value"])

    def _split_half(self, arr: np.ndarray):
        """
        Split array into two equal halves for convergence diagnostics.
        """
        half = len(arr) // 2
        return arr[:half], arr[half:]

    def _render_robustness_panels(
        self,
        df_mc: pd.DataFrame,
        human_dis: np.ndarray,
        llm_dis: np.ndarray,
        iterations: int,
    ):
        """
        Internal rendering engine for single-LLM robustness figure (Panels A, B, C).
        """
        import seaborn as sns
        import matplotlib.pyplot as plt

        sns.set_theme(style="whitegrid", font_scale=1.2)
        fig, axes = plt.subplots(1, 3, figsize=(22, 6))

        # ----- Panel A -----
        pA, pB = self._split_half(df_mc["p_value"].values)
        sns.kdeplot(pA, fill=True, ax=axes[0], label="First half", alpha=0.5, color="royalblue")
        sns.kdeplot(pB, fill=True, ax=axes[0], label="Second half", alpha=0.5, color="#333333")
        axes[0].axvline(0.05, linestyle="--", color="black")
        axes[0].set_title(
            "Convergence Diagnostics:\nMonte Carlo p-Value Distribution",
            fontweight="bold",
            fontsize=18,
        )
        axes[0].set_xlabel("p-Value", fontsize=14, fontweight="bold")
        axes[0].set_ylabel("Density", fontsize=14, fontweight="bold")
        axes[0].legend()

        # ----- Panel B -----
        dA, dB = self._split_half(df_mc["delta_mean"].values)
        sns.kdeplot(dA, fill=True, ax=axes[1], label="First half", alpha=0.5, color="royalblue")
        sns.kdeplot(dB, fill=True, ax=axes[1], label="Second half", alpha=0.5, color="#333333")
        axes[1].axvline(0, linestyle="-.", color="black")
        axes[1].set_title(
            "Convergence Diagnostics:\nMonte Carlo Δ Mean Disagreement",
            fontweight="bold",
            fontsize=18,
        )
        axes[1].set_xlabel("Δ Mean Disagreement", fontsize=14, fontweight="bold")
        axes[1].set_ylabel("Density", fontsize=14, fontweight="bold")
        axes[1].legend()

        # ----- Panel C -----
        sns.scatterplot(
            data=df_mc, x="delta_mean", y="p_value",
            alpha=0.3, s=100, color="orangered", ax=axes[2],
            label="Random judges"
        )

        # GPT-4 reference point
        delta_llm = np.mean(llm_dis) - np.mean(human_dis)
        p_val_llm = mannwhitneyu(llm_dis, human_dis, alternative="greater").pvalue

        axes[2].scatter(
            delta_llm, p_val_llm,
            color="lightgreen", edgecolor="black",
            s=120, marker="X", linewidth=1.2,
            label="LLM"
        )

        axes[2].axhline(0.05, linestyle="--", color="black")
        axes[2].axvline(0, linestyle="-.", color="gray")
        axes[2].set_title(
            f"Δ Mean vs p-Value: Monte Carlo Cloud vs LLM \n({iterations:,} samples)",
            fontweight="bold",
            fontsize=18,
        )
        axes[2].set_xlabel("Δ Mean", fontsize=14, fontweight="bold")
        axes[2].set_ylabel("p-Value", fontsize=14, fontweight="bold")
        axes[2].legend()

        plt.tight_layout()
        return fig

    def _render_robustness_panels_multi_llm(
        self,
        df_mc: pd.DataFrame,
        human_dis: np.ndarray,
        llm_results: dict[str, dict],
        iterations: int,
    ):
        """
        Internal rendering engine for multi-LLM robustness figure (Panels A, B, C).
        
        Parameters
        ----------
        df_mc : pd.DataFrame
            Monte Carlo simulation results with columns 'delta_mean' and 'p_value'.
        human_dis : np.ndarray
            Human-human disagreement array.
        llm_results : dict
            Dictionary mapping LLM names to their results:
            {name: {"delta": float, "p_value": float}}
        iterations : int
            Number of Monte Carlo iterations (for title).
        """
        import seaborn as sns
        import matplotlib.pyplot as plt

        sns.set_theme(style="whitegrid", font_scale=1.2)
        fig, axes = plt.subplots(1, 3, figsize=(22, 6))

        # ----- Panel A: p-Value Convergence -----
        pA, pB = self._split_half(df_mc["p_value"].values)
        sns.kdeplot(pA, fill=True, ax=axes[0], label="First half", alpha=0.5, color="royalblue")
        sns.kdeplot(pB, fill=True, ax=axes[0], label="Second half", alpha=0.5, color="#333333")
        axes[0].axvline(0.05, linestyle="--", color="black")
        axes[0].set_title(
            "Panel A: Convergence Diagnostics\nMonte Carlo p-Value Distribution",
            fontweight="bold",
            fontsize=18,
        )
        axes[0].set_xlabel("p-Value", fontsize=14, fontweight="bold")
        axes[0].set_ylabel("Density", fontsize=14, fontweight="bold")
        axes[0].legend()

        # ----- Panel B: Δ Mean Convergence -----
        dA, dB = self._split_half(df_mc["delta_mean"].values)
        sns.kdeplot(dA, fill=True, ax=axes[1], label="First half", alpha=0.5, color="royalblue")
        sns.kdeplot(dB, fill=True, ax=axes[1], label="Second half", alpha=0.5, color="#333333")
        axes[1].axvline(0, linestyle="-.", color="black")
        axes[1].set_title(
            "Panel B: Convergence Diagnostics\nMonte Carlo Δ Mean Disagreement",
            fontweight="bold",
            fontsize=18,
        )
        axes[1].set_xlabel("Δ Mean Disagreement", fontsize=14, fontweight="bold")
        axes[1].set_ylabel("Density", fontsize=14, fontweight="bold")
        axes[1].legend()

        # ----- Panel C: Monte Carlo Cloud + Multiple LLMs -----
        sns.scatterplot(
            data=df_mc, x="delta_mean", y="p_value",
            alpha=0.3, s=100, color="orangered", ax=axes[2],
            label="Random judges"
        )

        # Color palette for multiple LLMs
        llm_colors = ["#2ecc71", "#3498db", "#9b59b6", "#e74c3c", "#f39c12", "#1abc9c"]
        llm_markers = ["X", "o", "s", "D", "^", "v"]

        for idx, (llm_name, results) in enumerate(llm_results.items()):
            color = llm_colors[idx % len(llm_colors)]
            marker = llm_markers[idx % len(llm_markers)]
            
            axes[2].scatter(
                results["delta"], results["p_value"],
                color=color, edgecolor="black",
                s=180, marker=marker, linewidth=1.5,
                label=llm_name, zorder=10
            )

        axes[2].axhline(0.05, linestyle="--", color="black")
        axes[2].axvline(0, linestyle="-.", color="gray")
        
        n_llms = len(llm_results)
        title_suffix = "LLMs" if n_llms > 1 else "LLM"
        axes[2].set_title(
            f"Panel C: Δ Mean vs p-Value\nMonte Carlo Cloud ({iterations:,} samples) vs {n_llms} {title_suffix}",
            fontweight="bold",
            fontsize=18,
        )
        axes[2].set_xlabel("Δ Mean", fontsize=14, fontweight="bold")
        axes[2].set_ylabel("p-Value", fontsize=14, fontweight="bold")
        axes[2].legend(loc="upper right")

        plt.tight_layout()
        return fig



    def plot_monte_carlo_robustness(
        self,
        llm_col: str,
        iterations: int = 25_000,
        save_path: str | None = None,
    ) -> None:
        """
        Run a Monte-Carlo robustness analysis of the LLM-as-a-judge evaluation.

        This procedure repeatedly re-initializes the Random Judge (via reseeding),
        recomputes its disagreement distribution with human annotators, and evaluates:

        • Panel A — Stability of p-value distributions across Monte-Carlo splits  
        • Panel B — Stability of Δ-mean disagreement across Monte-Carlo splits  
        • Panel C — Joint Δ-mean vs p-value scatter (random judges vs a fixed LLM)  

        These diagnostics reveal whether the evaluation is statistically robust
        to random fluctuations in the baseline (random) judge. If GPT-4o-mini (or
        another LLM) consistently falls outside the Monte-Carlo cloud, the
        performance is considered stable and “better-than-random” in a
        statistically meaningful way.

        This method uses your improved simulation logic:  
        - direct sampling of random judge scores  
        - vectorized disagreement computation  
        - convergence diagnostics via split-half KDE comparisons  

        Parameters
        ----------
        llm_col : str
            Name of the LLM judge column to evaluate.
        iterations : int, default=25_000
            Number of Monte-Carlo samples (random judges) to generate.
        save_path : str or None
            Optional file path to save the resulting 3-panel figure.
        """

        human_dis = self.compute_human_disagreements()
        llm_dis = self.compute_llm_human_disagreements(llm_col)

        df_mc = self._monte_carlo_random_judges(
            iterations=iterations,
            min_score=self.min_score,
            max_score=self.max_score,
            human_dis=human_dis,
        )

        fig = self._render_robustness_panels(
            df_mc=df_mc,
            human_dis=human_dis,
            llm_dis=llm_dis,
            iterations=iterations,
        )

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()

    def plot_monte_carlo_robustness_multi(
        self,
        llm_cols: list[str] | None = None,
        iterations: int = 25_000,
        save_path: str | None = None,
    ) -> None:
        """
        Run Monte-Carlo robustness analysis comparing multiple LLMs simultaneously.

        This method generates a single 3-panel figure where:
        
        • Panel A — Stability of p-value distributions across Monte-Carlo splits  
        • Panel B — Stability of Δ-mean disagreement across Monte-Carlo splits  
        • Panel C — Joint Δ-mean vs p-value scatter with ALL specified LLMs plotted
        
        This is more efficient than calling `plot_monte_carlo_robustness` in a loop
        because the Monte Carlo simulation only runs once, and all LLMs are compared
        in a single Panel C visualization.

        Parameters
        ----------
        llm_cols : list of str, optional
            Names of the LLM judge columns to evaluate. If None, uses all LLM columns
            provided during initialization (self.llm_cols).
        iterations : int, default=25_000
            Number of Monte-Carlo samples (random judges) to generate.
        save_path : str or None
            Optional file path to save the resulting 3-panel figure.

        Example
        -------
        ```python
        # Instead of looping:
        # for model in ['GPT', 'LLAMA', 'MISTRAL']:
        #     evaluator.plot_monte_carlo_robustness(f"{model}_relevance_as_a_judge")
        
        # Use this single call:
        evaluator.plot_monte_carlo_robustness_multi(
            llm_cols=['GPT_relevance_as_a_judge', 'LLAMA_relevance_as_a_judge', 'MISTRAL_relevance_as_a_judge'],
            iterations=33_000
        )
        ```
        """
        # Default to all LLM columns if none specified
        if llm_cols is None:
            llm_cols = self.llm_cols
        
        if not llm_cols:
            raise ValueError("❌ No LLM columns specified and no default llm_cols available.")

        # Validate all columns exist
        missing = [col for col in llm_cols if col not in self.df.columns]
        if missing:
            raise ValueError(f"❌ LLM columns not found in DataFrame: {missing}")

        if self.verbosity > 0:
            print(f"📊 Running Monte Carlo robustness analysis for {len(llm_cols)} LLMs...")
            for col in llm_cols:
                print(f"   • {col}")

        # Compute human disagreements (once)
        human_dis = self.compute_human_disagreements()

        # Run Monte Carlo simulation (once)
        df_mc = self._monte_carlo_random_judges(
            iterations=iterations,
            min_score=self.min_score,
            max_score=self.max_score,
            human_dis=human_dis,
        )

        # Compute results for each LLM
        llm_results = {}
        for llm_col in llm_cols:
            llm_dis = self.compute_llm_human_disagreements(llm_col)
            delta = np.mean(llm_dis) - np.mean(human_dis)
            p_val = mannwhitneyu(llm_dis, human_dis, alternative="greater").pvalue
            
            # Extract a clean display name from column name
            # e.g., "GPT_relevance_as_a_judge" -> "GPT"
            display_name = llm_col.split("_")[0] if "_" in llm_col else llm_col
            
            llm_results[display_name] = {
                "delta": delta,
                "p_value": p_val,
                "col": llm_col,
            }
            
            if self.verbosity > 0:
                print(f"   ✓ {display_name}: Δ={delta:.4f}, p={p_val:.4f}")

        # Plot all LLMs in single figure
        fig = self._render_robustness_panels_multi_llm(
            df_mc=df_mc,
            human_dis=human_dis,
            llm_results=llm_results,
            iterations=iterations,
        )

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            if self.verbosity > 0:
                print(f"✅ Figure saved → {save_path}")

        plt.show()
