import pandas as pd
import numpy as np
import random
from itertools import combinations
from typing import Optional
from scipy.stats import mannwhitneyu
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import seaborn as sns
sns.set_theme(style="whitegrid")
from joblib import Parallel, delayed
import multiprocessing




class LLMGoodEnough:
    """
    Is your selected LLM-as-a-judge good enough?

    This class evaluates whether an LLM's performance is "good enough" and hence
    suitable for automated evaluation tasks of other LLM-generated outputs.
    It does so by comparing absolute disagreement distributions between:

        - Human–Human judgments (diversity of opinion)
        - LLM–Human judgments

    These distributions are compared using a one-sided Mann–Whitney U test to
    assess whether the LLM deviates more from humans than humans deviate from
    each other.

    Randomness & Reproducibility
    ----------------------------
    All Monte Carlo and stability analyses in this class use NumPy's
    `np.random.default_rng` with explicitly managed RNG streams. We intentionally
    avoid `np.random.randint` and other global RNG calls in order to ensure:

        - reproducibility across runs given a fixed seed
        - independence of Monte Carlo draws (fresh random judges per iteration)
        - deterministic behavior under parallel execution

    Each sample size (percentage) is assigned its own RNG stream derived from
    the base seed, guaranteeing parallel-safe and reproducible results.

    Corresponding paper: https://arxiv.org/abs/--->>>ToBeAnnounced<<<---

    Example:
    ```python
    from core.llm_good_enough import LLMGoodEnough
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
        Set *global/legacy* RNG seeds for reproducibility.

        Notes
        -----
        - Seeds NumPy's legacy global RNG (`np.random.*`) and Python's `random`.
        - This does NOT affect `np.random.default_rng(...)` generators.
        - All Monte Carlo and stability analyses intentionally use local
        `default_rng` streams instead of `np.random.randint` to ensure:
            * reproducibility across runs
            * independence of draws
            * parallel-safe execution
        """
        np.random.seed(seed)
        random.seed(seed)

    
    def init_random_judge(
        self,
        min_score: int,
        max_score: int,
        seed: int | None = None
    ) -> np.ndarray:
        """
        Create a baseline random judge column.

        Important design choice
        -----------------------
        - If `seed` is None: use the instance seed (`self.seed`) so the baseline column
        is deterministic and reproducible for a fixed evaluator instance.
        - If `seed` is provided: override the instance seed for explicit control.

        Note
        ----
        This baseline column is mainly used for *static* plots (histograms/bar plots).
        For Monte Carlo / stability analyses we generate "fresh random judges"
        inside the loops, NOT by mutating this column.
        """
        use_seed = self.seed if seed is None else int(seed)
        rng = np.random.default_rng(use_seed)
        return rng.integers(min_score, max_score + 1, size=len(self.df))



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
    def compute_human_disagreements(self, df: pd.DataFrame | None = None) -> np.ndarray:
        """
        Compute the Human–Human disagreement distribution.

        For each item, returns absolute pairwise differences |H_i − H_j| across all
        human rater pairs (i < j), ignoring missing ratings.

        Parameters
        ----------
        df : pd.DataFrame or None
            Optional dataframe to compute on. Defaults to `self.df`.

        Returns
        -------
        np.ndarray
            Flattened array of absolute human–human disagreements.
        """
        df_use = self.df if df is None else df

        # Coerce to numeric (non-numeric -> NaN), then convert to numpy
        H = df_use[self.human_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)  # (n_items, n_humans)
        n_items, n_humans = H.shape

        if n_humans < 2:
            raise ValueError("❌ Need at least two human columns to compute disagreements.")

        # All unique (i < j) human rater pairs
        ii, jj = np.triu_indices(n_humans, k=1)

        out_chunks: list[np.ndarray] = []
        for i, j in zip(ii, jj):
            a = H[:, i]
            b = H[:, j]
            mask = (~np.isnan(a)) & (~np.isnan(b))
            if np.any(mask):
                out_chunks.append(np.abs(a[mask] - b[mask]))

        if not out_chunks:
            raise ValueError("❌ No valid human-human pairs found (check input data / missingness).")

        return np.concatenate(out_chunks).astype(float, copy=False)




    def compute_llm_human_disagreements(
        self,
        llm_col: str,
        df: pd.DataFrame | None = None,
    ) -> np.ndarray:
        """
        Compute the LLM–Human disagreement distribution.

        For each item, computes absolute differences |LLM − H_j| between the LLM
        rating and all available human ratings. Rows with missing LLM ratings
        are ignored; missing human ratings are skipped.

        Parameters
        ----------
        llm_col : str
            Column name of the LLM judge.
        df : pd.DataFrame or None
            Optional dataframe to compute on. Defaults to `self.df`.

        Returns
        -------
        np.ndarray
            Flattened array of absolute LLM–Human disagreements.
        """
        df_use = self.df if df is None else df

        if llm_col not in df_use.columns:
            raise ValueError(f"❌ LLM column not found in DataFrame: {llm_col}")

        human_matrix = df_use[self.human_cols].apply(pd.to_numeric, errors="coerce").to_numpy()
        llm_vec = pd.to_numeric(df_use[llm_col], errors="coerce").to_numpy()

        diffs = np.abs(human_matrix - llm_vec[:, None])
        mask = (~np.isnan(human_matrix)) & (~np.isnan(llm_vec)[:, None])

        out = diffs[mask]

        if out.size == 0:
            raise ValueError("❌ No valid LLM–human pairs found.")

        return out.astype(float, copy=False)



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

        Render histogram-based comparisons between human–human disagreement
        (distribution of human opinion diversity) and model–human disagreement
        (distribution of model deviation from humans).

        This function assumes that *disagreement magnitude is defined by the
        rating scale itself*, not by the observed data range. Therefore, when
        `bins` is not explicitly provided, histogram bins are constructed from
        the theoretical maximum disagreement:

            max_possible_disagreement = max_score − min_score

        This guarantees:
            • consistent binning across datasets and subsamples
            • comparability between human, LLM, and random-judge baselines
            • alignment with Monte Carlo and stability analyses

        This method is purely a rendering utility and does not perform any
        statistical inference beyond visualizing precomputed distributions.

        Parameters
        ----------
        model_disagreement_dict : dict
            Mapping from model name → disagreement array.
            Each array contains absolute differences between the model’s ratings
            and human ratings.

        human_human_disagreements : np.ndarray, optional
            Array of absolute human–human disagreements.
            If None, it is computed from the full dataset.

        bins : np.ndarray, optional
            Explicit histogram bin edges.
            If None, bins are constructed as:
                np.arange(0, (max_score − min_score) + 2)

        bar_width : float, default=0.35
            Width of histogram bars for side-by-side comparison.

        y_lim : float, default=0.6
            Upper limit for the probability density axis.

        save_path : str or None
            Optional path to save the rendered figure.

        Returns
        -------
        None
            This method renders and optionally saves a matplotlib figure.
        """

        if human_human_disagreements is None:
            human_human_disagreements = self.compute_human_disagreements()
        if bins is None:
            max_possible_disagreement = self.max_score - self.min_score
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

            # counts (how many samples in each distribution)
            n_blue = int(len(human_human_disagreements))   # human-human
            n_red  = int(len(data))                        # model-human

            if self.verbosity >= 2:
                print(f"[{display_name}] n_blue(HH)={n_blue:,} | n_red(MH)={n_red:,} | p={p_val:.4g}")


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

            ax.set_title(display_name, fontweight='bold', fontsize=21)
            ax.set_xlabel('Degree of Disagreement', fontsize=17, fontweight='bold')
            ax.set_ylabel('Probability', fontsize=17, fontweight='bold')
            ax.set_xticks(categories)
            ax.tick_params(axis='both', which='major', labelsize=16)
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
            ax.set_title(name, fontweight="bold", fontsize=21)
            ax.set_xlabel("Degree of Disagreement", fontsize=17, fontweight="bold")
            ax.set_ylabel("Probability", fontsize=17, fontweight="bold")
            ax.set_xticks(categories)
            ax.set_yticks(np.arange(0, 0.6, 0.1))
            ax.tick_params(axis="both", which="major", labelsize=16)
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
        df_items: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        INTERNAL: Monte-Carlo simulation of *fresh random judges*.

        Pattern A (parallel-safe RNG)
        -----------------------------
        Each Monte Carlo iteration uses its own independent RNG stream derived from a
        master SeedSequence (self.seed). This guarantees:
        - reproducibility for a fixed self.seed
        - independence between Monte Carlo draws
        - parallel-safety (iteration order / worker scheduling won't change results)

        For each iteration:
        1) Draw one random rating per item (row): R_i
        2) Compute Random–Human disagreements: |R_i - H_ij| for all non-missing H_ij
        3) Compute effect size: delta_mean = mean(Random–Human) - mean(Human–Human)
        4) Compute one-sided MWU p-value for Random–Human > Human–Human
        """

        # Preallocate results array
        results = np.empty((iterations, 2), dtype=float)

        # ---- ensure df_items is usable and consistent ----
        df_use = df_items.copy()

        # Enforce ">=2 human ratings" rule (recommended for stability / consistency)
        mask_2plus = df_use[self.human_cols].notna().sum(axis=1) >= 2
        df_use = df_use.loc[mask_2plus].copy()

        # Coerce human ratings to numeric consistently (matches your other methods)
        human_matrix = (
            df_use[self.human_cols]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(dtype=float)
        )
        n_items = human_matrix.shape[0]

        if n_items == 0:
            raise ValueError("❌ No valid items with ≥2 human ratings for Monte Carlo simulation.")

        # IMPORTANT: baseline must match df_use (not self.df)
        human_human_dis = self.compute_human_disagreements(df=df_use)
        mean_hh = float(np.mean(human_human_dis))

        # ----- Pattern A RNG: independent stream per iteration -----
        ss = np.random.SeedSequence(self.seed)
        child_seeds = ss.spawn(iterations)

        # Monte Carlo loop
        for i, child_ss in enumerate(child_seeds):
            rng_i = np.random.default_rng(child_ss)

            # 1) Fresh random rating per item (this is the "fresh random judge")
            r = rng_i.integers(min_score, max_score + 1, size=n_items).astype(float)

            # 2) Random–Human disagreements for all non-missing humans
            diffs = np.abs(human_matrix - r[:, None])          # (n_items, n_humans)
            random_human_dis = diffs[~np.isnan(human_matrix)]  # flatten valid entries only

            if random_human_dis.size == 0:
                results[i] = (np.nan, np.nan)
                continue

            # 3) Effect size
            delta = float(np.mean(random_human_dis) - mean_hh)

            # 4) Significance test
            p_val = float(
                mannwhitneyu(random_human_dis, human_human_dis, alternative="greater").pvalue
            )

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
    ) -> plt.Figure:
        """
        Internal rendering engine for single-LLM robustness figure (Panels A, B, C).

        Parameters
        ----------
        df_mc : pd.DataFrame
            Monte Carlo results with columns 'delta_mean' and 'p_value'.
        human_dis : np.ndarray
            Human–Human disagreement distribution.
        llm_dis : np.ndarray
            LLM–Human disagreement distribution.
        iterations : int
            Number of Monte Carlo iterations.
        
        Returns
        -------
        plt.Figure
            The rendered matplotlib figure with three panels.
        """


        sns.set_theme(style="whitegrid", font_scale=1.2)
        fig, axes = plt.subplots(1, 3, figsize=(22, 6))

        # ----- Panel A -----
        pA, pB = self._split_half(df_mc["p_value"].values)
        label_pA = f"First half\n(μ={np.mean(pA):.3f}, σ={np.std(pA):.3f})"
        label_pB = f"Second half\n(μ={np.mean(pB):.3f}, σ={np.std(pB):.3f})"
        sns.kdeplot(pA, fill=True, ax=axes[0], label=label_pA, alpha=0.5, color="royalblue")
        sns.kdeplot(pB, fill=True, ax=axes[0], label=label_pB, alpha=0.5, color="#333333")
        axes[0].axvline(0.05, linestyle="--", color="black")
        axes[0].set_title(
            "Monte Carlo p-Value Distribution",
            fontweight="bold",
            fontsize=21,
        )
        axes[0].set_xlabel("p-Value", fontsize=17, fontweight="bold")
        axes[0].set_ylabel("Density", fontsize=17, fontweight="bold")
        axes[0].tick_params(axis='both', which='major', labelsize=16)
        axes[0].legend(edgecolor="black", facecolor="white", framealpha=1, fontsize=17, loc="best")

        # ----- Panel B -----
        dA, dB = self._split_half(df_mc["delta_mean"].values)
        label_dA = f"First half\n(μ={np.mean(dA):.4f}, σ={np.std(dA):.4f})"
        label_dB = f"Second half\n(μ={np.mean(dB):.4f}, σ={np.std(dB):.4f})"
        sns.kdeplot(dA, fill=True, ax=axes[1], label=label_dA, alpha=0.5, color="royalblue")
        sns.kdeplot(dB, fill=True, ax=axes[1], label=label_dB, alpha=0.5, color="#333333")
        axes[1].axvline(0, linestyle="-.", color="black")
        axes[1].set_title(
            "Monte Carlo Δ Mean Disagreement Distribution",
            fontweight="bold",
            fontsize=21,
        )
        axes[1].set_xlabel("Δ Mean Disagreement", fontsize=17, fontweight="bold")
        axes[1].set_ylabel("Density", fontsize=17, fontweight="bold")
        axes[1].tick_params(axis='both', which='major', labelsize=16)
        axes[1].legend(edgecolor="black", facecolor="white", framealpha=1, fontsize=17, loc="best")

        # ----- Panel C -----
        sns.scatterplot(
            data=df_mc, x="delta_mean", y="p_value",
            alpha=0.3, s=100, color="orangered", ax=axes[2],
            label=f"Random judges\n(n={iterations:,})"
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
            "Monte Carlo Cloud vs LLM Judges",
            fontweight="bold",
            fontsize=21,
        )
        axes[2].set_xlabel("Δ Mean Disagreement", fontsize=17, fontweight="bold")
        axes[2].set_ylabel("p-Value", fontsize=17, fontweight="bold")
        axes[2].tick_params(axis='both', which='major', labelsize=16)
        axes[2].legend(edgecolor="black", facecolor="white", framealpha=1, fontsize=17)

        plt.tight_layout()
        return fig
    


    def _is_orangeish(self, color: str, threshold: float = 0.20) -> bool:
        rgb = np.array(mcolors.to_rgb(color))
        oranges = [
            np.array(mcolors.to_rgb("orange")),
            np.array(mcolors.to_rgb("orangered")),
        ]
        return any(np.linalg.norm(rgb - o) < threshold for o in oranges)



    def _make_llm_style_map(self, llm_names: list[str]) -> dict[str, dict]:
        """
        Assign styles so that:
        1) Initially: all markers AND colors are unique
        2) Colors are never reused until necessary
        3) Orange-like colors are excluded
        4) No identical (marker, color) pair can occur
        """

        # --- Strong, clearly distinguishable markers ---
        markers = ["X", "o", "s", "D", "^", "v", "P", "*", "<", ">", "h", "H", "p", "8"]

        # --- Start with high-quality categorical palettes ---
        colors = list(mcolors.TABLEAU_COLORS.values())
        colors += [mcolors.to_hex(c) for c in plt.get_cmap("tab20").colors]

        # Remove orange-like colors
        colors = [c for c in colors if not self._is_orangeish(c)]

        style_map = {}
        used_colors = set()
        used_pairs = set()

        # =========================
        # Phase 1: unique marker + unique color
        # =========================
        for name, marker, color in zip(llm_names, markers, colors):
            style_map[name] = {"marker": marker, "color": color}
            used_colors.add(color)
            used_pairs.add((marker, color))

        remaining = llm_names[len(style_map):]

        # =========================
        # Phase 2: reuse markers, still unique colors
        # =========================
        color_pool = [c for c in colors if c not in used_colors]
        marker_idx = 0

        for name in remaining:
            if not color_pool:
                break
            marker = markers[marker_idx % len(markers)]
            color = color_pool.pop(0)

            style_map[name] = {"marker": marker, "color": color}
            used_pairs.add((marker, color))
            used_colors.add(color)
            marker_idx += 1

        remaining = llm_names[len(style_map):]

        # =========================
        # Phase 3: last resort (reuse colors + markers)
        # =========================
        if remaining:
            for name in remaining:
                for marker in markers:
                    for color in colors:
                        if self._is_orangeish(color):
                            continue
                        if (marker, color) not in used_pairs:
                            style_map[name] = {"marker": marker, "color": color}
                            used_pairs.add((marker, color))
                            break
                    if name in style_map:
                        break

        return style_map



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
        sns.set_theme(style="whitegrid", font_scale=1.2)

        fig, axes = plt.subplots(1, 3, figsize=(22, 6))

        # ----- Panel A: p-Value Convergence -----
        pA, pB = self._split_half(df_mc["p_value"].values)
        label_pA = f"First half\n(μ={np.mean(pA):.3f}, σ={np.std(pA):.3f})"
        label_pB = f"Second half\n(μ={np.mean(pB):.3f}, σ={np.std(pB):.3f})"
        sns.kdeplot(pA, fill=True, ax=axes[0], label=label_pA, alpha=0.5, color="royalblue")
        sns.kdeplot(pB, fill=True, ax=axes[0], label=label_pB, alpha=0.5, color="#333333")
        axes[0].axvline(0.05, linestyle="--", color="black")
        axes[0].set_title(
            "Monte Carlo p-Value Distribution",
            fontweight="bold",
            fontsize=21,
        )
        axes[0].set_xlabel("p-Value", fontsize=17, fontweight="bold")
        axes[0].set_ylabel("Density", fontsize=17, fontweight="bold")
        axes[0].tick_params(axis='both', which='major', labelsize=16)
        axes[0].legend(edgecolor="black", facecolor="white", framealpha=1, fontsize=17, loc="best")

        # ----- Panel B: Δ Mean Convergence -----
        dA, dB = self._split_half(df_mc["delta_mean"].values)
        label_dA = f"First half\n(μ={np.mean(dA):.4f}, σ={np.std(dA):.4f})"
        label_dB = f"Second half\n(μ={np.mean(dB):.4f}, σ={np.std(dB):.4f})"
        sns.kdeplot(dA, fill=True, ax=axes[1], label=label_dA, alpha=0.5, color="royalblue")
        sns.kdeplot(dB, fill=True, ax=axes[1], label=label_dB, alpha=0.5, color="#333333")
        axes[1].axvline(0, linestyle="-.", color="black")
        axes[1].set_title(
            "Monte Carlo Δ Mean Disagreement Distribution",
            fontweight="bold",
            fontsize=21,
        )
        axes[1].set_xlabel("Δ Mean Disagreement", fontsize=17, fontweight="bold")
        axes[1].set_ylabel("Density", fontsize=17, fontweight="bold")
        axes[1].tick_params(axis='both', which='major', labelsize=16)
        axes[1].legend(edgecolor="black", facecolor="white", framealpha=1, fontsize=17, loc="best")

        # ----- Panel C: Monte Carlo Cloud + Multiple LLMs -----
        sns.scatterplot(
            data=df_mc, 
            x="delta_mean",
            y="p_value",
            alpha=0.3,
            s=100,
            color="orangered", ax=axes[2],
            label=f"Random judges\n(n={iterations:,})"
        )

        llm_names = list(llm_results.keys())
        style_map = self._make_llm_style_map(llm_names)

        for llm_col, results in llm_results.items():
            label = results["display_name"]
            style = style_map[llm_col]

            axes[2].scatter(
                results["delta"],
                results["p_value"],
                color=style["color"],
                marker=style["marker"],
                edgecolor="black",
                s=180,
                linewidth=1.5,
                label=label,
                zorder=10,
            )

        axes[2].axhline(0.05, linestyle="--", color="black")
        axes[2].axvline(0, linestyle="-.", color="gray")

        axes[2].set_title(
            "Monte Carlo Cloud vs LLM Judges",
            fontweight="bold",
            fontsize=21,
        )
        axes[2].set_xlabel("Δ Mean Disagreement", fontsize=17, fontweight="bold")
        axes[2].set_ylabel("p-Value", fontsize=17, fontweight="bold")
        axes[2].tick_params(axis='both', which='major', labelsize=16)
        axes[2].legend(edgecolor="black", facecolor="white", framealpha=1, fontsize=17)

        plt.tight_layout()
        return fig



    def plot_monte_carlo_robustness(
        self,
        llm_col: str,
        iterations: int = 25_000,
        save_path: str | None = None,
        df_items: pd.DataFrame | None = None,
    ) -> None:
        """
        Run a Monte-Carlo robustness analysis of the LLM-as-a-judge evaluation.

        This procedure repeatedly draws fresh random ratings each iteration from one RNG stream seeded with self.seed,
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
        df_use = self.df if df_items is None else df_items

        llm_dis = self.compute_llm_human_disagreements(llm_col, df=df_use)

        df_mc = self._monte_carlo_random_judges(
            iterations=iterations,
            min_score=self.min_score,
            max_score=self.max_score,
            df_items=df_use,             # item-level dataframe
        )

        human_dis = self.compute_human_disagreements(df=df_use)
        fig = self._render_robustness_panels(
            df_mc=df_mc,
            human_dis=human_dis,
            llm_dis=llm_dis,
            iterations=iterations,
        )

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")



    def count_pvalue_samples(
        self,
        iterations: int = 25_000,
        threshold: float = 0.05,
    ) -> dict:
        """
        Count how many Monte Carlo p-values fall above/below a significance threshold.

        This is useful for quantifying Panel A of the robustness analysis:
        "How often does a random judge appear statistically indistinguishable from humans?"

        Parameters
        ----------
        iterations : int, default=25_000
            Number of Monte Carlo samples (random judges) to generate.
        threshold : float, default=0.05
            Significance threshold (typically 0.05).

        Returns
        -------
        dict
            Dictionary with keys:
            - 'total': total number of samples
            - 'above': count of p-values > threshold
            - 'below': count of p-values ≤ threshold  
            - 'above_pct': percentage above threshold
            - 'below_pct': percentage below threshold
            - 'threshold': the threshold used

        Example
        -------
        ```python
        counts = judge.count_pvalue_samples(iterations=25_000, threshold=0.05)
        print(f"Samples with p > 0.05: {counts['above']:,} ({counts['above_pct']:.2f}%)")
        print(f"Samples with p ≤ 0.05: {counts['below']:,} ({counts['below_pct']:.2f}%)")
        ```
        """
        df_mc = self._monte_carlo_random_judges(
            iterations=iterations,
            min_score=self.min_score,
            max_score=self.max_score,
            df_items=self.df,
        )

        total = len(df_mc)
        above = int((df_mc['p_value'] > threshold).sum())
        below = int((df_mc['p_value'] <= threshold).sum())

        return {
            'total': total,
            'above': above,
            'below': below,
            'above_pct': above / total * 100,
            'below_pct': below / total * 100,
            'threshold': threshold,
        }



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
            df_items=self.df,
        )


        # Compute results for each LLM
        llm_results = {}
        for llm_col in llm_cols:
            llm_dis = self.compute_llm_human_disagreements(llm_col)
            delta = np.mean(llm_dis) - np.mean(human_dis)
            p_val = mannwhitneyu(llm_dis, human_dis, alternative="greater").pvalue
            
            llm_results[llm_col] = {
                "delta": delta,
                "p_value": p_val,
                "display_name": self._clean_model_name(llm_col),
            }

            
            if self.verbosity > 0:
                print(f"   ✓ {llm_results[llm_col]['display_name']}: Δ={delta:.4f}, p={p_val:.4f}")
        
        # de-duplicate display names for legend clarity
        seen = {}
        for col, d in llm_results.items():
            name = d["display_name"]
            seen[name] = seen.get(name, 0) + 1

        if any(v > 1 for v in seen.values()):
            for col, d in llm_results.items():
                if seen[d["display_name"]] > 1:
                    d["display_name"] = col.replace("_as_a_judge", "")


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



    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #~~~~~~~~~~~~~~~ Human Stability Analysis ~~~~~~~~~~~~~~~#
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    @staticmethod
    def _has_converged(
        mean_A: float,
        mean_B: float,
        convergence_threshold: float,
        relative_convergence: bool,
    ) -> bool:
        """
        Check if split-half acceptance rates have converged.

        This method compares two halves of the collected decisions to determine
        if the acceptance rate estimate has stabilized. Convergence indicates
        that additional iterations are unlikely to significantly change the result.

        Parameters
        ----------
        mean_A : float
            Mean acceptance rate of the first half of decisions.
        mean_B : float
            Mean acceptance rate of the second half of decisions.
        convergence_threshold : float
            Threshold for determining convergence.
            - If `relative_convergence=False`: absolute difference threshold
            - If `relative_convergence=True`: proportion of the mean
        relative_convergence : bool
            Whether to use relative or absolute convergence criterion.

        Returns
        -------
        bool
            True if the split-half difference is below the effective threshold.

        Notes
        -----
        **Absolute mode:**
            Converged when ``|A - B| < convergence_threshold``

        **Relative mode:**
            Converged when ``|A - B| < max(threshold * mean(A,B), threshold/10)``
            
            The floor (threshold/10) prevents issues when acceptance rate ≈ 0,
            which would otherwise cause premature convergence.
        """
        diff = abs(mean_A - mean_B)

        if relative_convergence:
            # Relative: threshold as proportion of current mean
            mean_both = (mean_A + mean_B) / 2
            # Floor prevents premature convergence when mean ≈ 0
            floor = convergence_threshold / 10
            effective_threshold = max(convergence_threshold * mean_both, floor)
        else:
            # Absolute: fixed threshold regardless of acceptance rate
            effective_threshold = convergence_threshold

        return diff < effective_threshold



    def _compute_stability_for_percentage(
        self,
        p: int,
        df: pd.DataFrame,
        min_iterations: int,
        max_iterations: int,
        check_interval: int,
        convergence_threshold: float,
        relative_convergence: bool,
        seed: int,
    ) -> tuple[int, float, bool, int]:
        """
        Compute acceptance rate stability for a single sample percentage.

        This is the core worker function that runs the Monte Carlo simulation
        for one sample size. It can be called in parallel across different
        percentages for significant speedup.

        Parameters
        ----------
        p : int
            Percentage of data to sample (e.g., 10 means 10% of rows).
        df : pd.DataFrame
            The full dataset to sample from.
        min_iterations : int
            Minimum iterations before checking convergence.
        max_iterations : int
            Hard cap on iterations to prevent infinite loops.
        check_interval : int
            Check convergence every N iterations (after min_iterations).
        convergence_threshold : float
            Threshold for split-half convergence (see `_has_converged`).
        relative_convergence : bool
            Whether to use relative or absolute convergence criterion.
        seed : int
            Base seed for reproducibility. Actual seed is `seed + p` to ensure
            different percentages get different (but reproducible) random streams.

        Returns
        -------
        tuple of (int, float, bool, int)
            - percentage: the input percentage p
            - mean_acceptance: acceptance rate (proportion of p > 0.05 outcomes)
            - converged: whether the simulation converged before max_iterations
            - n_iterations: total number of iterations performed
        """
        # RNG stream is fixed per (seed, percentage) for reproducibility.
        # IMPORTANT: Each loop iteration represents ONE Monte Carlo draw:
        #   (1) bootstrap-sample items
        #   (2) generate a FRESH random judge (new pseudo_vals vector)
        # This means: within one percentage p we create a new random judge per iteration,
        # rather than re-using a single fixed RANDOM_as_a_judge column.
        #
        # We intentionally use `default_rng` instead of `np.random.randint`
        # so that:
        #   - each iteration draws an independent random judge
        #   - results are reproducible across runs
        #   - parallel execution does not affect randomness

        rng = np.random.default_rng(seed + p)
        n_rows = max(1, int(len(df) * (p / 100)))

        decisions: list[bool] = []
        iteration_count = 0
        converged = False

        while iteration_count < max_iterations:
            # --- 1) Bootstrap subsample rows ---
            sample = df.sample(
                n_rows,
                replace=True,
                random_state=int(rng.integers(0, 2**32 - 1)),
            )

            # --- 2) Compute human–human disagreements ---
            try:
                human_dis = self.compute_human_disagreements(df=sample)
            except ValueError:
                # No valid human-human pairs in this sample
                # hence the iteration is "wasted" in terms of contributing a decision, 
                # but it still consumes one unit of your max_iterations budget
                iteration_count += 1
                continue

            # --- 3) Random–Human disagreements (fresh random judge each iteration) ---
            human_matrix = (
                sample[self.human_cols]
                .apply(pd.to_numeric, errors="coerce")
                .to_numpy(dtype=float)
            )                                               # (n_items, n_humans)
            pseudo_vals = rng.integers(self.min_score, self.max_score + 1, size=human_matrix.shape[0]).astype(float)

            diffs = np.abs(human_matrix - pseudo_vals[:, None])      # (n_items, n_humans)
            pseudo_dis = diffs[~np.isnan(human_matrix)]              # flatten valid entries only

            if pseudo_dis.size == 0:
                iteration_count += 1
                continue

            # --- 4) Mann-Whitney U test ---
            # H0: random-human disagreements ≤ human-human disagreements
            # Ha: random-human disagreements > human-human disagreements (one-sided)
            # Accept random judge as "human-like" if p > 0.05 (fail to reject H0)
            p_val = mannwhitneyu(pseudo_dis, human_dis, alternative="greater").pvalue
            decisions.append(p_val > 0.05)
            iteration_count += 1

            # --- 5) Periodic convergence check ---
            n_decisions = len(decisions)
            if n_decisions >= min_iterations and n_decisions % check_interval == 0:

                decisions_arr = np.asarray(decisions, dtype=float)
                half = len(decisions_arr) // 2

                # Guard: need at keastb 2 decisions to split meaningfully
                if half == 0:
                    continue

                mean_A = float(np.mean(decisions_arr[:half]))
                mean_B = float(np.mean(decisions_arr[half:]))

                if self._has_converged(
                    mean_A,
                    mean_B,
                    convergence_threshold,
                    relative_convergence,
                ):
                    converged = True
                    break

        # Compute final acceptance rate
        mean_acceptance = float(np.mean(decisions)) if decisions else np.nan
        return p, mean_acceptance, converged, iteration_count



    def _run_percentage_loop(
        self,
        percentages: list[int],
        *,
        parallel: bool,
        n_jobs: int | None,
        **worker_kwargs,
    ) -> list[tuple[int, float, bool, int]]:
        """
        Execute stability analysis across all sample percentages.

        This method orchestrates the stability computation, either sequentially
        or in parallel across multiple CPU cores.

        Parameters
        ----------
        percentages : list of int
            List of sample percentages to evaluate (e.g., [5, 10, 20, ...]).
        parallel : bool
            If True, distribute percentage computations across CPU cores.
        n_jobs : int or None
            Number of parallel jobs. If None and parallel=True, uses
            ``cpu_count() - 1`` to leave one core free for system tasks.
        **worker_kwargs
            Additional keyword arguments passed to `_compute_stability_for_percentage`.
            Typically includes: df, min_iterations, max_iterations,
            check_interval, convergence_threshold, relative_convergence, seed.

        Returns
        -------
        list of tuple
            List of (percentage, mean_acceptance, converged, n_iterations) tuples,
            one for each input percentage.

        Notes
        -----
        Uses joblib's "loky" backend for robust parallel execution that handles
        process cleanup gracefully. Each percentage gets a unique seed derived
        from the base seed, ensuring reproducible results regardless of
        parallel vs sequential execution.
        """
        if parallel:
            # Default to all cores minus one (leave headroom for system)
            if n_jobs is None:
                n_jobs = max(1, multiprocessing.cpu_count() - 1)

            # Parallel execution using joblib
            results = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(self._compute_stability_for_percentage)(
                    p=p,
                    **worker_kwargs,
                )
                for p in percentages
            )
        else:
            # Sequential execution
            results = [
                self._compute_stability_for_percentage(
                    p=p,
                    **worker_kwargs,
                )
                for p in percentages
            ]

        return results

    def _compute_llm_stability_for_percentage(
        self,
        p: int,
        df: pd.DataFrame,
        llm_col: str,
        min_iterations: int,
        max_iterations: int,
        check_interval: int,
        convergence_threshold: float,
        relative_convergence: bool,
        seed: int,
    ) -> tuple[int, float, float, float, float, bool, int]:
        """
        Compute LLM acceptance-rate stability + effect size (Δ mean) for one sample percentage.

        For each bootstrap iteration on a sample of size p%:
        - compute Human–Human disagreement distribution
        - compute LLM–Human disagreement distribution for `llm_col`
        - run one-sided MWU test (alternative='greater') for LLM–Human > Human–Human
        - record decision: accepted if p_value > 0.05 (fail-to-reject)
        - record delta_mean = mean(LLM–Human) - mean(Human–Human)

        Convergence is detected via split-half acceptance rate stability (same as random-judge stability).
        """
        if llm_col not in df.columns:
            raise ValueError(f"❌ LLM column not found in DataFrame: {llm_col}")

        rng = np.random.default_rng(seed + p)
        n_rows = max(1, int(len(df) * (p / 100)))

        decisions: list[bool] = []
        deltas: list[float] = []
        iteration_count = 0
        converged = False

        while iteration_count < max_iterations:
            # --- 1) Bootstrap subsample rows ---
            sample = df.sample(
                n_rows,
                replace=True,
                random_state=int(rng.integers(0, 2**32 - 1)),
            )

            # --- 2) Human–Human disagreements (canonical method) ---
            try:
            
                human_dis = self.compute_human_disagreements(df=sample)
            except ValueError:
                # No valid human-human pairs in this sample
                iteration_count += 1
                continue


            # --- 3) LLM–Human disagreements (canonical method) ---
            try:
                llm_dis = self.compute_llm_human_disagreements(llm_col=llm_col, df=sample)
            except ValueError:
                # No valid LLM-human pairs in this sample
                iteration_count += 1
                continue


            # --- 4) MWU test + record decision ---
            p_val = mannwhitneyu(llm_dis, human_dis, alternative="greater").pvalue
            decisions.append(p_val > 0.05)
            deltas.append(float(np.mean(llm_dis) - np.mean(human_dis)))
            iteration_count += 1

            # --- 5) Convergence check (split-half acceptance rate) ---
            n_decisions = len(decisions)
            if n_decisions >= min_iterations and n_decisions % check_interval == 0:
                decisions_arr = np.asarray(decisions, dtype=float)
                half = len(decisions_arr) // 2

                # Guard: need at least 2 decisions to split meaningfully
                if half == 0:
                    continue

                mean_A = float(np.mean(decisions_arr[:half]))
                mean_B = float(np.mean(decisions_arr[half:]))

                if self._has_converged(
                    mean_A,
                    mean_B,
                    convergence_threshold,
                    relative_convergence,
                ):
                    converged = True
                    break

        mean_acceptance = float(np.mean(decisions)) if decisions else np.nan
        mean_delta = float(np.mean(deltas)) if deltas else np.nan
        p10_delta = float(np.percentile(deltas, 10)) if deltas else np.nan
        p90_delta = float(np.percentile(deltas, 90)) if deltas else np.nan
        return p, mean_acceptance, mean_delta, p10_delta, p90_delta, converged, iteration_count


    def _run_percentage_loop_llm(
        self,
        percentages: list[int],
        *,
        parallel: bool,
        n_jobs: int | None,
        **worker_kwargs,
    ) -> list[tuple[int, float, float, float, float, bool, int]]:
        """
        Execute LLM stability analysis across all sample percentages.
        """
        if parallel:
            if n_jobs is None:
                n_jobs = max(1, multiprocessing.cpu_count() - 1)

            results = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(self._compute_llm_stability_for_percentage)(
                    p=p,
                    **worker_kwargs,
                )
                for p in percentages
            )
        else:
            results = [
                self._compute_llm_stability_for_percentage(
                    p=p,
                    **worker_kwargs,
                )
                for p in percentages
            ]

        return results



    def plot_human_stability_analysis(
        self,
        percentages: list[int] = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        min_iterations: int = 200,
        max_iterations: int = 10_000,
        check_interval: int = 100,
        convergence_threshold: float = 0.01,
        relative_convergence: bool = True,
        parallel: bool = False,
        n_jobs: int | None = None,
        save_path: str | None = None,
        show_iteration_counts: bool = True,
    ) -> None:
        """
        Test how stable acceptance rates are across different sample sizes.

        Uses **adaptive sampling**: iterates until the acceptance rate converges
        (split-half difference < threshold) rather than fixed iteration counts.
        This is statistically sound because smaller samples (noisier) automatically
        get more iterations, while larger samples stop early once stable.

        Reasoning
        ---------
        - Smaller sample percentages (e.g., 5%) are inherently noisier and may
          need more iterations to stabilize.
        - Larger percentages converge faster, so fixed iterations waste compute.
        - Showing iteration counts provides insight into the "difficulty" of
          each sample size.

        **Algorithm:**
            For each percentage p:
                1. Bootstrap-sample p% of rows
                2. Generate a fresh random judge (new random ratings each iteration)
                3. Compute human–human and random-judge–human disagreements
                4. Run MWU test (accepted if p > 0.05)
                5. Repeat until split-half acceptance rates converge or max_iterations

        **Acceptance rate**
            = number of times the random judge is accepted as human-like / total iterations

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
            - Flat high (~1.0) → judge is consistently human-like across all sample sizes.
            - Flat low (~0.0) → judge is clearly bad, rejected even with tiny samples.
            - Uphill → suspicious, investigate data quality or methodology.
            - Erratic (many red points) → high variance, may need more data.

        Parameters
        ----------
        percentages : list of int
            Percentages of data to sample (default: 5% to 100% in steps).
        min_iterations : int
            Minimum iterations before checking convergence.
        max_iterations : int
            Hard cap to prevent infinite loops.
        check_interval : int
            Check convergence every N iterations (after min_iterations).
        convergence_threshold : float
            Stop when split-half difference in acceptance rate is below this.
            Interpretation depends on `relative_convergence`:
            - If False (absolute): threshold is a fixed value (e.g., 0.01 = 1 pp)
            - If True (relative): threshold is a proportion (e.g., 0.05 = 5% of mean)
        relative_convergence : bool, default=True
            Whether to use relative or absolute convergence criterion.
            -------------------------------------------------------------------------
            | **Relative (True, recommended):**                                     |
            |    Stop when ``|mean_A - mean_B| < threshold * mean(mean_A, mean_B)``.|
            |    Ensures proportional stability: demands tighter precision at low   |
            |    rates and allows more slack at high rates.                         |
            |    A floor of ``threshold / 10`` prevents issues when mean ≈ 0.       |
            |-----------------------------------------------------------------------|
            | **Absolute (False):**                                                 |
            |    Stop when ``|mean_A - mean_B| < threshold``.                       |
            |    Uses fixed precision regardless of acceptance rate.                |
            -------------------------------------------------------------------------
        parallel : bool, default=False
            If True, each percentage is evaluated on a separate CPU core.
            Provides significant speedup for large analyses. Statistical behavior
            is IDENTICAL to the serial version (reproducible via seed).
        n_jobs : int or None
            Number of parallel jobs. If None, uses ``cpu_count() - 1``.
            Only relevant when ``parallel=True``.
        save_path : str or None
            Path to save the figure (e.g., "stability_plot.png").
        show_iteration_counts : bool
            Annotate iteration counts above each scatter point.
        """
        # Prepare data
        df = self.df.reset_index(drop=True)

        # Run Monte Carlo simulations (serial or parallel)
        results = self._run_percentage_loop(
            percentages=percentages,
            parallel=parallel,
            n_jobs=n_jobs,
            df=df,
            min_iterations=min_iterations,
            max_iterations=max_iterations,
            check_interval=check_interval,
            convergence_threshold=convergence_threshold,
            relative_convergence=relative_convergence,
            seed=self.seed,
        )

        # Unpack results for plotting
        perc, acc, conv, iters = zip(*results)

        # =====================================================================
        # Plotting
        # =====================================================================

        fig, ax = plt.subplots(figsize=(12, 7))

        # Main line plot connecting all points
        ax.plot(
            perc, acc,
            marker="o",
            linewidth=2.5,
            markersize=10,
            markeredgecolor="white",
            markeredgewidth=1.5,
            zorder=1,
        )

        # Scatter points colored by convergence status
        for p_val, a, c, n_iter in results:
            color = "springgreen" if c else "orangered"
            ax.scatter(
                p_val, a,
                color=color,
                s=140,
                zorder=2,
                edgecolor="black",
                linewidth=1,
            )

            # Annotate iteration count (always above scatter point)
            if show_iteration_counts and not np.isnan(a):
                ax.annotate(
                    f"{n_iter:,}",
                    xy=(p_val, a),
                    xytext=(0, 14),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=14,
                    color="dimgray",
                    fontweight="bold",
                )

        # Reference lines
        ax.axhline(0.5, linestyle="--", color="gray", alpha=0.6)
        ax.axhline(0.0, linestyle="-", color="black", alpha=0.2, linewidth=1)

        # Legend for convergence status
        legend_elements = [
            Line2D(
                [0], [0],
                marker="o",
                color="w",
                markerfacecolor="springgreen",
                markersize=10,
                markeredgecolor="black",
                label="Converged",
            ),
            Line2D(
                [0], [0],
                marker="o",
                color="w",
                markerfacecolor="orangered",
                markersize=10,
                markeredgecolor="black",
                label="Max iterations reached",
            ),
        ]
        ax.legend(
            handles=legend_elements,
            loc="upper right",
            fontsize=14,
            edgecolor="black",
            facecolor="white",
            framealpha=1,
        )

        # Labels and styling
        ax.set_title(
            "Sample Size vs Random Judge Acceptance",
            fontsize=21,
            fontweight="bold",
        )
        ax.set_xlabel("Percentage of Data Sampled", fontsize=17, fontweight="bold")
        ax.set_ylabel("Acceptance Rate (p > 0.05)", fontsize=17, fontweight="bold")
        ax.tick_params(axis="both", which="major", labelsize=16)
        ax.set_ylim(-0.12, 1.15)  # Room for low values + annotations
        ax.set_xlim(min(perc) - 3, max(perc) + 3)
        ax.grid(alpha=0.3)

        # Add note explaining what annotations represent (bottom left corner)
        if show_iteration_counts:
            ax.text(
                0.02, 0.02,
                "Annotations = number of iterations until convergence",
                transform=ax.transAxes,
                fontsize=14,
                color="gray",
                style="italic",
            )

        plt.tight_layout()

        # Save figure if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            if self.verbosity > 0:
                print(f"✅ Figure saved → {save_path}")

        plt.show()


    def plot_llm_stability_analysis(
        self,
        llm_col: str,
        percentages: list[int] = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        min_iterations: int = 200,
        max_iterations: int = 10_000,
        check_interval: int = 100,
        convergence_threshold: float = 0.01,
        relative_convergence: bool = True,
        parallel: bool = False,
        n_jobs: int | None = None,
        save_path: str | None = None,
        show_iteration_counts: bool = True,
    ) -> None:
        """
        LLM Stability Analysis (sample size vs acceptance + effect size).

        Mirrors `plot_human_stability_analysis()` but replaces the random judge with a
        *specific LLM judge column* and produces a 2-panel figure:

        - Panel A: acceptance rate vs sample percentage (accepted if p > 0.05)
        - Panel B: Δ mean disagreement vs sample percentage
          Δ = mean(LLM–Human) − mean(Human–Human)

        Interpretation:
        - stable high acceptance near 100% + Δ≈0 suggests the LLM is robustly “good enough”
        - acceptance dropping with sample size (often with Δ>0) indicates the LLM is not robustly human-like;
          small-sample acceptance may have been due to low power.
        """
        if llm_col not in self.df.columns:
            raise ValueError(f"❌ LLM column not found in DataFrame: {llm_col}")

        df = self.df.reset_index(drop=True)

        results = self._run_percentage_loop_llm(
            percentages=percentages,
            parallel=parallel,
            n_jobs=n_jobs,
            df=df,
            llm_col=llm_col,
            min_iterations=min_iterations,
            max_iterations=max_iterations,
            check_interval=check_interval,
            convergence_threshold=convergence_threshold,
            relative_convergence=relative_convergence,
            seed=self.seed,
        )

        perc, acc, delta, delta_p10, delta_p90, conv, iters = zip(*results)


        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

        # -------------------------
        # Panel A: Acceptance rate
        # -------------------------
        ax1.plot(
            perc, acc,
            marker="o",
            linewidth=2.5,
            markersize=10,
            markeredgecolor="white",
            markeredgewidth=1.5,
            zorder=1,
        )

        for p_val, a, _delta, _p10, _p90, c, n_iter in results:
            color = "springgreen" if c else "orangered"
            ax1.scatter(
                p_val, a,
                color=color,
                s=140,
                zorder=2,
                edgecolor="black",
                linewidth=1,
            )

            if show_iteration_counts and not np.isnan(a):
                ax1.annotate(
                    f"{n_iter:,}",
                    xy=(p_val, a),
                    xytext=(0, 14),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=12,
                    color="dimgray",
                    fontweight="bold",
                )

        ax1.axhline(0.5, linestyle="--", color="gray", alpha=0.6)
        ax1.axhline(0.0, linestyle="-", color="black", alpha=0.2, linewidth=1)
        ax1.set_title("LLM Acceptance Rate vs Sample Size", fontsize=18, fontweight="bold")
        ax1.set_xlabel("Percentage of Data Sampled", fontsize=14, fontweight="bold")
        ax1.set_ylabel("Acceptance Rate (p > 0.05)", fontsize=14, fontweight="bold")
        ax1.tick_params(axis="both", which="major", labelsize=12)
        ax1.set_ylim(-0.12, 1.15)
        ax1.set_xlim(min(perc) - 3, max(perc) + 3)
        ax1.grid(alpha=0.3)

        legend_elements = [
            Line2D(
                [0], [0],
                marker="o",
                color="w",
                markerfacecolor="springgreen",
                markersize=10,
                markeredgecolor="black",
                label="Converged",
            ),
            Line2D(
                [0], [0],
                marker="o",
                color="w",
                markerfacecolor="orangered",
                markersize=10,
                markeredgecolor="black",
                label="Max iterations reached",
            ),
        ]
        ax1.legend(
            handles=legend_elements,
            loc="best",
            fontsize=12,
            edgecolor="black",
            facecolor="white",
            framealpha=1,
        )

        if show_iteration_counts:
            ax1.text(
                0.02, 0.02,
                "Annotations = iterations until convergence",
                transform=ax1.transAxes,
                fontsize=11,
                color="gray",
                style="italic",
            )

        # -------------------------
        # Panel B: Δ mean
        # -------------------------
        ax2.plot(
            perc, delta,
            marker="o",
            linewidth=2.5,
            markersize=10,
            markeredgecolor="white",
            markeredgewidth=1.5,
            zorder=1,
        )

        # Show variability band (10th–90th percentile) across bootstrap runs
        ax2.fill_between(
            perc,
            delta_p10,
            delta_p90,
            alpha=0.2,
            color="steelblue",
            label="P10–P90 of Δ across bootstrap runs",
            zorder=0,
        )

        for p_val, d, c in zip(perc, delta, conv):
            color = "springgreen" if c else "orangered"
            ax2.scatter(
                p_val, d,
                color=color,
                s=140,
                zorder=2,
                edgecolor="black",
                linewidth=1,
            )

        ax2.axhline(0.0, linestyle="-.", color="black", alpha=0.7)
        ax2.set_title("Δ Mean Disagreement vs Sample Size", fontsize=18, fontweight="bold")
        ax2.set_xlabel("Percentage of Data Sampled", fontsize=14, fontweight="bold")
        ax2.set_ylabel(
            "Relative Disagreement (LLM vs Humans)",
            fontsize=14,
            fontweight="bold",
        )
        ax2.tick_params(axis="both", which="major", labelsize=12)
        ax2.set_xlim(min(perc) - 3, max(perc) + 3)
        ax2.grid(alpha=0.3)
        ax2.legend(loc="best", fontsize=11, edgecolor="black", facecolor="white", framealpha=1)

        fig.suptitle(f"LLM Stability Analysis — {self._clean_model_name(llm_col)}", fontsize=20, fontweight="bold")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            if self.verbosity > 0:
                print(f"✅ Figure saved → {save_path}")

        plt.show()


    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #~~~~~~~~~~~~~~~ Seed Robustness Analysis ~~~~~~~~~~~~~#
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    def _run_stability_single_seed(
        self,
        seed: int,
        percentages: list[int],
        min_iterations: int,
        max_iterations: int,
        check_interval: int,
        convergence_threshold: float,
        relative_convergence: bool = True,   # ✅ add
    ) -> dict[int, tuple[float, bool, int]]:
        """
        Internal: Run stability analysis for a single seed without plotting.
        Returns dict mapping percentage -> (acceptance_rate, converged, n_iterations).
        """
        df = self.df.reset_index(drop=True)

        results: dict[int, tuple[float, bool, int]] = {}

        for p in percentages:
            n_rows = max(1, int(len(df) * (p / 100)))
            decisions: list[bool] = []
            converged = False
            iteration_count = 0

            rng = np.random.default_rng(seed + p)

            while iteration_count < max_iterations:
                sample = df.sample(
                    n_rows,
                    replace=True,
                    random_state=int(rng.integers(0, 2**32 - 1)),
                )

                # 1) Human–Human disagreements
                try:
                    human_dis = self.compute_human_disagreements(df=sample)
                except ValueError:
                    iteration_count += 1
                    continue

                # 2) Random–Human disagreements (fresh random judge)
                human_matrix = (
                    sample[self.human_cols]
                    .apply(pd.to_numeric, errors="coerce")
                    .to_numpy(dtype=float)
                )

                pseudo_vals = rng.integers(self.min_score, self.max_score + 1, size=human_matrix.shape[0]).astype(float)

                pseudo_dis = np.abs(human_matrix - pseudo_vals[:, None])
                pseudo_dis = pseudo_dis[~np.isnan(human_matrix)]
                if pseudo_dis.size == 0:
                    iteration_count += 1
                    continue

                # 3) MWU + decision
                p_val = mannwhitneyu(pseudo_dis, human_dis, alternative="greater").pvalue
                decisions.append(p_val > 0.05)
                iteration_count += 1

                # 4) Convergence check 
                n_decisions = len(decisions)
                if n_decisions >= min_iterations and n_decisions % check_interval == 0:

                    decisions_arr = np.asarray(decisions, dtype=float)
                    half = len(decisions_arr) // 2
                    if half == 0:
                        continue

                    mean_A = float(np.mean(decisions_arr[:half]))
                    mean_B = float(np.mean(decisions_arr[half:]))

                    if self._has_converged(
                        mean_A,
                        mean_B,
                        convergence_threshold,
                        relative_convergence,
                    ):
                        converged = True
                        break

            acc = float(np.mean(decisions)) if decisions else np.nan
            results[p] = (acc, converged, iteration_count)

        return results


    def plot_human_stability_seed_robustness(
        self,
        n_seeds: int = 20,
        percentages: list[int] = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        min_iterations: int = 200,
        max_iterations: int = 5000,
        check_interval: int = 100,
        convergence_threshold: float = 0.01,
        relative_convergence: bool = True,
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
                relative_convergence=relative_convergence,
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
        
        # Mean line with larger markers for visibility at edges
        ax.plot(percentages, means, 'o-', color='steelblue', linewidth=2.5, markersize=10, 
                markeredgecolor='white', markeredgewidth=1.5, label='Mean acceptance rate', zorder=3)
        
        # Annotate convergence rate (adaptive position: above if high, below if low)
        for p, mean, conv_count in zip(percentages, means, [convergence_rates[p] for p in percentages]):
            if not np.isnan(mean):
                conv_pct = conv_count / n_seeds * 100
                # Put annotation below if mean is low, above if mean is high
                if mean < 0.15:
                    offset_y, va = -18, 'top'
                else:
                    offset_y, va = 15, 'bottom'
                ax.annotate(
                    f'{conv_pct:.0f}%',
                    xy=(p, mean),
                    xytext=(0, offset_y),
                    textcoords='offset points',
                    ha='center',
                    va=va,
                    fontsize=9,
                    color='darkgreen' if conv_pct > 80 else 'darkorange',
                    fontweight='bold',
                )
        
        # Reference lines
        ax.axhline(0.5, linestyle='--', color='gray', alpha=0.6, label='50% threshold')
        ax.axhline(0.0, linestyle='-', color='black', alpha=0.2, linewidth=1)  # Zero line for visibility
        
        ax.set_title(
            f"Seed Sensitivity Analysis\n(Mean ± {int(confidence_level*100)}% CI across {n_seeds} seeds)",
            fontsize=21, fontweight='bold'
        )
        ax.set_xlabel("Percentage of Data Sampled", fontsize=17, fontweight='bold')
        ax.set_ylabel("Acceptance Rate", fontsize=17, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.set_ylim(-0.12, 1.15)  # More room at bottom for low values + annotations
        ax.set_xlim(min(percentages) - 3, max(percentages) + 3)
        ax.legend(loc='upper right', fontsize=14, edgecolor="black", facecolor="white", framealpha=1)
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
