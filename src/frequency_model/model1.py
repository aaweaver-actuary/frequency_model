import numpy as np
import pandas as pd
import pymc as pm
from pathlib import Path

DATA = Path("../..").resolve() / "data"
BETA_PRIOR_SIGMA = 0.5
BETA_RATIO_PRIOR_SIGMA = 0.1


def build_triangle(df: pd.DataFrame, loss_type: str) -> pd.DataFrame:
    """Select the columns needed for the triangle and pivot the data to a triangle shape.

    Parameters
    ----------
    df : pd.DataFrame
        The input data. Must have columns `acc_year`, `dev_lag`, and `loss_type`.
    loss_type : str
        The column name for the loss data. Must be a column in `df`.

    Returns
    -------
    pd.DataFrame
        A triangle-shaped DataFrame with `acc_year` as the index and `dev_lag` as the columns.

    Raises
    ------
    KeyError
        If `loss_type` is not a column in `df`.
    """
    if loss_type not in df.columns:
        raise KeyError(f"{loss_type} not in columns of df")

    return df[["acc_year", "dev_lag", loss_type]].pivot_table(
        index="acc_year",
        columns="dev_lag",
        values=loss_type,
        aggfunc="sum",
        fill_value=0,
    )


def triangle_of_prior_betas(triangle: pd.DataFrame) -> np.ndarray:
    """Calculate the prior beta parameters for the logit-normal distribution.

    Parameters
    ----------
    triangle : pd.DataFrame
        The triangle of data, with accident year as the index and development lag as the columns.

    Returns
    -------
    np.ndarray
        The prior beta parameters for the logit-normal distribution.
    """
    n_acc, n_dev = triangle.shape
    output = []
    for i in range(n_dev - 1):
        output.append(
            triangle.iloc[: (n_acc - i), i + 1].sum()
            / triangle.iloc[: (n_acc - i), i].sum()
        )

    real = np.divide(1, np.array(list(reversed(np.cumprod(list(reversed(output)))))))
    real = np.append(real, np.sqrt(real[-1]))

    return np.log(real / (1 - real))


def get_est_sd_of_obs(triangle: pd.DataFrame) -> float:
    """Calculate the estimated standard deviation of the observations.

    Parameters
    ----------
    triangle : pd.DataFrame
        The triangle of data, with accident year as the index and development lag as the columns.

    Returns
    -------
    float
        The estimated standard deviation of the observations.
    """
    df = (triangle - (triangle.sum().sum() / triangle.count().sum())) ** 2 / (
        triangle.sum().sum() / triangle.count().sum()
    )

    return np.sqrt(df.sum().sum() / df.count().sum())


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the data for the model.

    Parameters
    ----------
    df : pd.DataFrame
        The input data. Must have columns `incurred_loss`, `reported_loss`, and `paid_loss`.

    Returns
    -------
    pd.DataFrame
        The cleaned data with additional columns for the model.
    """
    df["paid_rpt_ratio"] = df["paid_loss"] / df["reported_loss"]
    df["loss_ratio"] = df["incurred_loss"] / df["ep"]
    df["log_loss_ratio"] = df["loss_ratio"].apply(lambda x: np.log(x) if x > 0 else 0)
    df["ave__log_loss_ratio"] = df["log_loss_ratio"].mean()
    df["std__log_loss_ratio"] = df["log_loss_ratio"].std()
    return df


def get_current_diagonal(df: pd.DataFrame) -> pd.DataFrame:
    """Get the current diagonal of the triangle.

    Parameters
    ----------
    df : pd.DataFrame
        The input data. Must have columns `acc_year`, `dev_lag`, and `ep`.

    Returns
    -------
    pd.DataFrame
        The current diagonal of the triangle.
    """
    max_ay = df["acc_year"].max()
    cur_diagonal = df.loc[df.acc_year + df.dev_lag - 1 == max_ay]
    cur_diagonal["log_prem"] = cur_diagonal["ep"].apply(
        lambda x: np.log(x) if x > 0 else 0
    )
    return cur_diagonal


def get_log_lr(df: pd.DataFrame) -> pd.Series:
    """Get the log loss ratio from the data.

    Parameters
    ----------
    df : pd.DataFrame
        The input data. Must have a column `log_loss_ratio` and `acc_year`.

    Returns
    -------
    pd.Series
        The log loss ratio.
    """
    cur_diag = get_current_diagonal(df)
    return cur_diag.set_index("acc_year").log_loss_ratio


def get_model(df: pd.DataFrame) -> pm.Model:
    df = clean_data(df)
    cur_diagonal = get_current_diagonal(df)
    log_lr = get_log_lr(df)

    obs_paid_loss = build_triangle("paid_loss")
    obs_reported_loss = build_triangle("reported_loss")
    obs_paid_reported_ratio = build_triangle("paid_rpt_ratio")

    prior_beta_rpt_loss = triangle_of_prior_betas(obs_reported_loss)
    prior_beta_ratio = triangle_of_prior_betas(obs_paid_reported_ratio)

    N_YEARS = df.acc_year.nunique()
    N_PERIODS = df.dev_lag.nunique()

    # Build the PyMC model
    with pm.Model() as model:
        # Priors for log expected loss ratio by accident year and derived ultimate loss
        log_elr__centered = pm.Normal("log_elr__centered", mu=0, sigma=1, shape=N_YEARS)
        log_elr = pm.Deterministic(
            "log_elr", log_elr__centered * log_lr.std() + log_lr.mean()
        )
        # log_elr = pm.Normal('log_elr', mu=log_lr.mean(), sigma=log_lr.std(), shape=N_YEARS)
        log_premium = cur_diagonal["log_prem"].to_numpy()
        log_ultimate = log_premium + log_elr

        # Expected ultimate loss by accident year is the sampled ELR times the premium
        # This is the ultimate loss for each accident year, and is the same for both paid and
        # reported losses
        expected_ultimate_loss = pm.Deterministic(
            "expected_ultimate_loss", np.exp(log_ultimate).reshape((N_YEARS, 1))
        )

        # Beta parameters sampled from a logit-normal distribution
        beta_rpt__centered = pm.Normal(
            "beta_rpt__centered", mu=0, sigma=1, shape=N_PERIODS
        )
        beta_ratio__centered = pm.Normal(
            "beta_ratio__centered", mu=0, sigma=1, shape=N_PERIODS
        )

        beta_rpt__logit = pm.Deterministic(
            "beta_rpt__logit",
            beta_rpt__centered * BETA_PRIOR_SIGMA + prior_beta_rpt_loss,
        )
        beta_ratio__logit = pm.Deterministic(
            "beta_ratio__logit",
            beta_ratio__centered * BETA_RATIO_PRIOR_SIGMA + prior_beta_ratio,
        )

        # This establishes the dependency of the paid loss beta on the reported loss beta
        beta_paid__logit = pm.Deterministic(
            "beta_paid__logit", beta_rpt__logit + beta_ratio__logit
        )

        # Beta parameters come from transforming the logit-normal parameters from above
        beta_rpt = pm.Deterministic("beta_rpt", pm.math.sigmoid(beta_rpt__logit))
        beta_paid = pm.Deterministic("beta_paid", pm.math.sigmoid(beta_paid__logit))

        # Expected loss for each cell in the triangle, for paid, reported, and the ratio
        # is the product of the expected ultimate loss and the beta parameter for that cell
        # and loss type
        expected_paid_loss = expected_ultimate_loss.reshape(
            (N_YEARS, 1)
        ) * beta_paid.reshape((1, N_PERIODS))
        expected_rpt_loss = expected_ultimate_loss.reshape(
            (N_YEARS, 1)
        ) * beta_rpt.reshape((1, N_PERIODS))

        # Observation noise (standard deviations)
        sigma_paid = pm.HalfNormal("sigma_paid", sigma=350, shape=(1, N_PERIODS))
        sigma_rpt = pm.HalfNormal("sigma_rpt", sigma=300, shape=(1, N_PERIODS))
        # sigma_ratio = pm.HalfNormal('sigma_ratio', sigma=0.25, shape=(1, N_PERIODS))

        # Shape and rate parameters for the gamma likelihood
        shape_paid = expected_paid_loss**2 / sigma_paid**2
        shape_rpt = expected_rpt_loss**2 / sigma_rpt**2
        rate_paid = expected_paid_loss / sigma_paid**2
        rate_rpt = expected_rpt_loss / sigma_rpt**2

        # Observed data likelihood -- assume a gamma distribution for the likelihood
        _paid_obs = pm.Gamma(
            "paid_obs", alpha=shape_paid, beta=rate_paid, observed=obs_paid_loss
        )
        _rpt_obs = pm.Gamma(
            "rpt_obs", alpha=shape_rpt, beta=rate_rpt, observed=obs_reported_loss
        )

        # Sampling from the posterior
        trace = pm.sample(1000, tune=2000, target_accept=0.975, random_seed=42)

    return model, trace
