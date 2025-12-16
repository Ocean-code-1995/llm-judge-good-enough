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
    @staticmethod
    def _clean_model_name(name: str) -> str:
        """
        Extract a clean display name from a column name.
        
        Examples:
            "GPT_relevance_as_a_judge" -> "GPT"
            "LLAMA_as_a_judge" -> "LLAMA"
            "GPT-4o-mini_coherence_as_a_judge" -> "GPT-4o-mini"
            "Random Judge" -> "Random Judge"
        """
        # Keep special names as-is
        if name in ("Random Judge", "RANDOM_as_a_judge"):
            return "Random Judge"
        
        # Remove "_as_a_judge" suffix and extract model name (first part before metric)
        name = name.replace("_as_a_judge", "")
        
        # If there's still an underscore, take the first part (model name)
        # But be careful with model names like "GPT-4o-mini" that don't have underscores
        parts = name.split("_")
        if len(parts) > 1:
            # Check if second part looks like a metric name
            metrics = {"informativeness", "relevance", "fluency", "coherence", "quality", "accuracy"}
            if parts[-1].lower() in metrics:
                return "_".join(parts[:-1])
        
        return parts[0] if parts else name

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

        for ax, (raw_name, data) in zip(axes, model_disagreement_dict.items()):
            # Clean up model name for display
            display_name = self._clean_model_name(raw_name)
            
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

            ax.set_title(display_name, fontweight='bold', fontsize=20)
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
        bar_width: float = 0.35,
        bins: np.ndarray = None,
        save_path: str = None,
    ) -> None:
        """
        Public API method.
        Plots a grid comparing all LLM judges + random judge against humans.

        Automatically:
        - computes human-human disagreements
        - computes each LLM's disagreements
        - adds Random Judge baseline
        - delegates rendering to the internal grid plotter

        Parameters
        ----------
        y_lim : float, default=0.6
            Upper limit of the y-axis (probability).
        save_path : str, optional
            Path to save the figure. Format inferred from extension.
        bar_width : float, default=0.35
            Width of the histogram bars.
        bins : np.ndarray, optional
            Custom bins for the histogram. If None, bins are inferred from 
            the score range (0 to max_possible_disagreement + 1).
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
            bins=bins,
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
            display_name = self._clean_model_name(llm_col)
            
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


    def plot_human_stability_analysis(
        self,
        percentages: list[int] = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        min_iterations: int = 200,
        max_iterations: int = 10000,
        check_interval: int = 100,
        convergence_threshold: float = 0.01,
        save_path: str | None = None,
        show_iteration_counts: bool = True,
    ) -> None:
        """
        Test how stable acceptance rates are across different sample sizes.
        
        Uses **adaptive sampling**: iterates until the acceptance rate converges 
        (split-half difference < threshold) rather than fixed iteration counts.
        This is statistically sound because smaller samples (noisier) automatically
        get more iterations, while larger samples stop early once stable.

        Reasoning:
        ---------
           - Smaller sample percentages (e.g., 5%) are inherently noisier and may need more iterations to stabilize.
           - Larger percentages converge faster, so we're wasting compute with fixed iterations
           - Showing iteration counts provides insight into the "difficulty" of each sample size

        **Algorithm:**
            For each percentage p:
            1. Bootstrap-sample p% of rows
            2. Compute human–human and random-judge–human disagreements
            3. Run MWU test (accepted if p > 0.05)
            4. Repeat until split-half acceptance rates converge or max_iterations
        
        **Acceptance rate** 
           = number of times the random judge is accepted as human-like / total number of iterations
        
        **Plot shows:**
            - X-axis: sample percentage, Y-axis: acceptance rate
            - Green scatter = converged, Red = max iterations reached
            - Annotations show iteration counts (reveals "difficulty" per sample size)

        **Interpretation:**
            The MWU test gains statistical power as sample size increases.
            - Small samples (5-10%): High acceptance rate → not enough power to detect 
              that random is worse than humans (Type II error / false negative).
            - Large samples (50-100%): Low acceptance rate → sufficient power to 
              reliably reject the random judge as non-human-like.
            
            This is a sanity check: a random judge *should* be rejected with enough 
            data. The downhill trend confirms the methodology is working correctly.
        
        **Trend patterns:**
            - Downhill → expected for random/bad judges. Power increases, rejection reliable.
            - Flat high (~1.0) → judge is consistently human-like across all sample sizes (good judge).
            - Flat low (~0.0) → judge is clearly bad, rejected even with tiny samples.
            - Uphill → suspicious, investigate data quality or methodology.
            - Erratic (many red points) → high variance, may need more data.

        Parameters
        ----------
        percentages : list of int
            Percentages of data to sample.
        min_iterations : int
            Minimum iterations before checking convergence.
        max_iterations : int
            Hard cap to prevent infinite loops.
        check_interval : int
            Check convergence every N iterations (after min_iterations).
        convergence_threshold : float
            Stop when split-half difference in acceptance rate is below this.
        save_path : str or None
            Path to save the figure.
        show_iteration_counts : bool
            Annotate iteration counts above each scatter point.
        """

        import matplotlib.pyplot as plt
        import numpy as np
        from scipy.stats import mannwhitneyu

        # Reseed for reproducibility across repeated calls
        self._set_global_seed(self.seed)

        human_cols = self.human_cols
        rand_col = "RANDOM_as_a_judge"
        df = self.df.reset_index(drop=True)

        # results: (percentage, mean_acceptance, converged_flag, n_iterations)
        results = []

        for p in percentages:
            n_rows = max(1, int(len(df) * (p / 100)))
            decisions = []
            converged = False
            iteration_count = 0

            if self.verbosity > 0:
                print(f"📊 Sampling {p}% of data ({n_rows} rows)...", end=" ")

            while iteration_count < max_iterations:
                # --- 1) Bootstrap subsample rows ---
                sample = df.sample(n_rows, replace=True)

                # --- 2) Compute human–human disagreements ---
                human_dis = []
                for _, row in sample[human_cols].iterrows():
                    vals = row.dropna().to_numpy()
                    if len(vals) >= 2:
                        diffs = np.abs(vals[:, None] - vals[None, :])[np.triu_indices(len(vals), k=1)]
                        human_dis.extend(diffs)

                human_dis = np.array(human_dis)
                if len(human_dis) == 0:
                    iteration_count += 1
                    continue

                # --- 3) Random-judge–human disagreements ---
                pseudo_vals = sample[rand_col].to_numpy()
                human_matrix = sample[human_cols].to_numpy()

                # mask rows where random judge is NaN (unlikely)
                mask = ~np.isnan(pseudo_vals)
                pseudo_vals = pseudo_vals[mask]
                human_sub = human_matrix[mask]

                pseudo_dis = []
                for pj, row_vals in zip(pseudo_vals, human_sub):
                    row_vals = row_vals[~np.isnan(row_vals)]
                    if len(row_vals) > 0:
                        pseudo_dis.extend(np.abs(row_vals - pj))

                pseudo_dis = np.array(pseudo_dis)
                if len(pseudo_dis) == 0:
                    iteration_count += 1
                    continue

                # --- 4) MWU test ---
                p_val = mannwhitneyu(
                    pseudo_dis, human_dis, alternative="greater"
                ).pvalue

                decisions.append(1 if p_val > 0.05 else 0)
                iteration_count += 1

                # --- 5) Check convergence periodically ---
                if (iteration_count >= min_iterations and 
                    iteration_count % check_interval == 0 and 
                    len(decisions) >= min_iterations):
                    
                    decisions_arr = np.array(decisions)
                    half = len(decisions_arr) // 2
                    mean_A = decisions_arr[:half].mean()
                    mean_B = decisions_arr[half:].mean()
                    
                    if abs(mean_A - mean_B) < convergence_threshold:
                        converged = True
                        break

            # --- 6) Handle empty decisions ---
            if len(decisions) == 0:
                results.append((p, np.nan, False, iteration_count))
                if self.verbosity > 0:
                    print(f"⚠️ No valid iterations")
                continue

            mean_acceptance = np.mean(decisions)
            results.append((p, mean_acceptance, converged, iteration_count))
            
            if self.verbosity > 0:
                status = "✓ converged" if converged else "⚠ max reached"
                print(f"{status} at {iteration_count:,} iterations (acceptance: {mean_acceptance:.3f})")

        # --- Plotting ---
        perc, acc, conv, iters = zip(*results)

        fig, ax = plt.subplots(figsize=(12, 7))
        ax.plot(perc, acc, marker="o", linewidth=2, zorder=1)

        for p, a, c, n_iter in results:
            color = "green" if c else "red"
            ax.scatter(p, a, color=color, s=120, zorder=2, edgecolor="black", linewidth=0.5)
            
            # Annotate iteration count above each point
            if show_iteration_counts and not np.isnan(a):
                ax.annotate(
                    f"{n_iter:,}",
                    xy=(p, a),
                    xytext=(0, 12),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=12,
                    color="dimgray",
                    fontweight="bold",
                )

        ax.axhline(0.5, linestyle="--", color="gray", alpha=0.6)
        
        # Add legend for convergence status
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
                   markersize=10, markeredgecolor='black', label='Converged'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                   markersize=10, markeredgecolor='black', label='Max iterations reached'),
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=11)
        
        ax.set_title(
            "Human Stability Test Using Random Judge\n(Adaptive Sampling Until Convergence)",
            fontsize=18, fontweight="bold"
        )
        ax.set_xlabel("Percentage of Data Sampled", fontsize=14)
        ax.set_ylabel("Acceptance Rate (p > 0.05)", fontsize=14)
        ax.set_ylim(0, 1.1)  # Extra space for annotations
        ax.set_xlim(min(perc) - 3, max(perc) + 3)
        ax.grid(alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            if self.verbosity > 0:
                print(f"✅ Figure saved → {save_path}")

        plt.show()


    def _run_stability_single_seed(
        self,
        seed: int,
        percentages: list[int],
        min_iterations: int,
        max_iterations: int,
        check_interval: int,
        convergence_threshold: float,
    ) -> dict:
        """
        Internal: Run stability analysis for a single seed without plotting.
        Returns dict mapping percentage -> (acceptance_rate, converged, n_iterations).
        """
        from scipy.stats import mannwhitneyu
        
        # Set seed
        self._set_global_seed(seed)
        
        human_cols = self.human_cols
        rand_col = "RANDOM_as_a_judge"
        df = self.df.reset_index(drop=True)
        
        results = {}
        
        for p in percentages:
            n_rows = max(1, int(len(df) * (p / 100)))
            decisions = []
            converged = False
            iteration_count = 0
            
            while iteration_count < max_iterations:
                sample = df.sample(n_rows, replace=True)
                
                # Compute human-human disagreements
                human_dis = []
                for _, row in sample[human_cols].iterrows():
                    vals = row.dropna().to_numpy()
                    if len(vals) >= 2:
                        diffs = np.abs(vals[:, None] - vals[None, :])[np.triu_indices(len(vals), k=1)]
                        human_dis.extend(diffs)
                
                human_dis = np.array(human_dis)
                if len(human_dis) == 0:
                    iteration_count += 1
                    continue
                
                # Random-judge-human disagreements
                pseudo_vals = sample[rand_col].to_numpy()
                human_matrix = sample[human_cols].to_numpy()
                mask = ~np.isnan(pseudo_vals)
                pseudo_vals = pseudo_vals[mask]
                human_sub = human_matrix[mask]
                
                pseudo_dis = []
                for pj, row_vals in zip(pseudo_vals, human_sub):
                    row_vals = row_vals[~np.isnan(row_vals)]
                    if len(row_vals) > 0:
                        pseudo_dis.extend(np.abs(row_vals - pj))
                
                pseudo_dis = np.array(pseudo_dis)
                if len(pseudo_dis) == 0:
                    iteration_count += 1
                    continue
                
                p_val = mannwhitneyu(pseudo_dis, human_dis, alternative="greater").pvalue
                decisions.append(1 if p_val > 0.05 else 0)
                iteration_count += 1
                
                # Check convergence
                if (iteration_count >= min_iterations and 
                    iteration_count % check_interval == 0 and 
                    len(decisions) >= min_iterations):
                    
                    decisions_arr = np.array(decisions)
                    half = len(decisions_arr) // 2
                    mean_A = decisions_arr[:half].mean()
                    mean_B = decisions_arr[half:].mean()
                    
                    if abs(mean_A - mean_B) < convergence_threshold:
                        converged = True
                        break
            
            if len(decisions) > 0:
                results[p] = (np.mean(decisions), converged, iteration_count)
            else:
                results[p] = (np.nan, False, iteration_count)
        
        return results


    def plot_human_stability_seed_robustness(
        self,
        n_seeds: int = 20,
        percentages: list[int] = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        min_iterations: int = 200,
        max_iterations: int = 5000,
        check_interval: int = 100,
        convergence_threshold: float = 0.01,
        confidence_level: float = 0.95,
        save_path: str | None = None,
    ) -> None:
        """
        Seed sensitivity analysis: run stability test across multiple seeds.
        
        Shows how robust the acceptance rate estimates are to random initialization.
        Plots mean acceptance rate with confidence intervals across seeds.

        **Interpretation:**
            - Tight CI bands → conclusions are robust across seeds, does not depend on the random seed
            - Wide CI bands → results are sensitive to random initialization
            - If the downhill trend is consistent across seeds → methodology is reliable

        Parameters
        ----------
        n_seeds : int
            Number of different seeds to test.
        percentages : list of int
            Percentages of data to sample.
        min_iterations : int
            Minimum iterations before checking convergence (per seed).
        max_iterations : int
            Maximum iterations per seed (lower than single-seed analysis for speed).
        check_interval : int
            Check convergence every N iterations.
        convergence_threshold : float
            Convergence threshold for split-half difference.
        confidence_level : float
            Confidence level for CI bands (default 95%).
        save_path : str or None
            Path to save the figure.
        """
        import matplotlib.pyplot as plt
        from scipy import stats
        
        if self.verbosity > 0:
            print(f"🔄 Running seed sensitivity analysis with {n_seeds} seeds...")
        
        # Generate random seeds (convert to Python int for random.seed compatibility)
        rng = np.random.default_rng(self.seed)
        seeds = [int(s) for s in rng.integers(0, 2**31, size=n_seeds)]
        
        # Collect results across seeds
        all_results = {p: [] for p in percentages}
        convergence_rates = {p: 0 for p in percentages}
        
        for i, seed in enumerate(seeds):
            if self.verbosity > 0:
                print(f"  Seed {i+1}/{n_seeds} ({seed})...", end=" ")
            
            result = self._run_stability_single_seed(
                seed=seed,
                percentages=percentages,
                min_iterations=min_iterations,
                max_iterations=max_iterations,
                check_interval=check_interval,
                convergence_threshold=convergence_threshold,
            )
            
            for p in percentages:
                acc, conv, n_iter = result[p]
                if not np.isnan(acc):
                    all_results[p].append(acc)
                    if conv:
                        convergence_rates[p] += 1
            
            if self.verbosity > 0:
                print("done")
        
        # Compute statistics
        means = []
        ci_lows = []
        ci_highs = []
        
        for p in percentages:
            values = np.array(all_results[p])
            if len(values) > 1:
                mean = np.mean(values)
                sem = stats.sem(values)
                # Handle edge case: SEM = 0 (all values identical) → CI collapses to mean
                if sem > 0:
                    ci = stats.t.interval(confidence_level, len(values)-1, loc=mean, scale=sem)
                    ci_lows.append(ci[0])
                    ci_highs.append(ci[1])
                else:
                    ci_lows.append(mean)
                    ci_highs.append(mean)
                means.append(mean)
            else:
                means.append(np.nan)
                ci_lows.append(np.nan)
                ci_highs.append(np.nan)
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # CI band
        ax.fill_between(
            percentages, ci_lows, ci_highs,
            alpha=0.3, color='steelblue', label=f'{int(confidence_level*100)}% CI across {n_seeds} seeds'
        )
        
        # Mean line
        ax.plot(percentages, means, 'o-', color='steelblue', linewidth=2, markersize=8, label='Mean acceptance rate')
        
        # Annotate convergence rate
        for p, mean, conv_count in zip(percentages, means, [convergence_rates[p] for p in percentages]):
            if not np.isnan(mean):
                conv_pct = conv_count / n_seeds * 100
                ax.annotate(
                    f'{conv_pct:.0f}%',
                    xy=(p, mean),
                    xytext=(0, 15),
                    textcoords='offset points',
                    ha='center',
                    fontsize=9,
                    color='darkgreen' if conv_pct > 80 else 'darkorange',
                    fontweight='bold',
                )
        
        ax.axhline(0.5, linestyle='--', color='gray', alpha=0.6)
        ax.set_title(
            f"Seed Sensitivity Analysis\n(Mean ± {int(confidence_level*100)}% CI across {n_seeds} seeds)",
            fontsize=18, fontweight='bold'
        )
        ax.set_xlabel("Percentage of Data Sampled", fontsize=14)
        ax.set_ylabel("Acceptance Rate", fontsize=14)
        ax.set_ylim(-0.05, 1.15)
        ax.set_xlim(min(percentages) - 3, max(percentages) + 3)
        ax.legend(loc='upper right', fontsize=11)
        ax.grid(alpha=0.3)
        
        # Add note about annotations
        ax.text(
            0.02, 0.02,
            "Annotations = % of seeds that converged",
            transform=ax.transAxes,
            fontsize=10, color='gray', style='italic'
        )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            if self.verbosity > 0:
                print(f"✅ Figure saved → {save_path}")
        
        plt.show()
        
        # Restore original seed
        self._set_global_seed(self.seed)
