import pandas as pd
import duckdb
from pathlib import Path

ROOT_FOLDER = Path(".").resolve()
DATA_DIR = ROOT_FOLDER / "data"
DATA_DIR.mkdir(exist_ok=True)

OTHER_LIABILITY_LINK = (
    "https://www.casact.org/sites/default/files/2021-04/othliab_pos.csv"
)

OLD_COLS = [
    "GRCODE",
    "GRNAME",
    "AccidentYear",
    "DevelopmentYear",
    "DevelopmentLag",
    "IncurLoss_h1",
    "CumPaidLoss_h1",
    "BulkLoss_h1",
    "EarnedPremDIR_h1",
    "EarnedPremCeded_h1",
    "EarnedPremNet_h1",
    "Single",
    "PostedReserve97_h1",
]

NEW_COLS = [
    "naic_code",
    "company_name",
    "acc_year",
    "dev_year",
    "dev_lag",
    "incurred_loss",
    "cumulative_paid_loss",
    "bulk_loss",
    "direct_ep",
    "ceded_ep",
    "net_ep",
    "is_single_entity",
    "ye_1997_reserves",
]

COLS_TO_KEEP = [
    "company_name",
    "acc_year",
    "dev_year",
    "dev_lag",
    "incurred_loss",
    "cumulative_paid_loss",
    "bulk_loss",
    "net_ep",
    "ye_1997_reserves",
]


def get_other_liability_data():
    return pd.read_csv(OTHER_LIABILITY_LINK)


def rename_columns(df, old=OLD_COLS, new=NEW_COLS):
    return df.rename(columns=dict(zip(old, new)))


def keep_columns(df, cols=COLS_TO_KEEP):
    return df[cols]


list_of_funcs = [rename_columns, keep_columns]


def main():
    other_liability_data = get_other_liability_data()

    for func in list_of_funcs:
        other_liability_data = func(other_liability_data)

    other_liability_data.to_csv(DATA_DIR / "other_liability_data.csv", index=False)


if __name__ == "__main__":
    main()

    data_loc = (DATA_DIR / "other_liability_data.csv").as_posix()

    with duckdb.connect(":memory:") as conn:
        df = conn.execute(f"""
            WITH 
                     
            raw AS (
                FROM read_csv('{data_loc}')
            ),

            unique_ep_rows AS (
                SELECT DISTINCT
                    company_name,
                    acc_year,
                    net_ep
                
                FROM raw
            ),

            aggregated_ep AS (
                SELECT
                    acc_year,
                    SUM(net_ep) AS ep
                FROM unique_ep_rows
                GROUP BY acc_year
            ),

            unique_loss_rows AS (
                SELECT DISTINCT
                    company_name,
                    acc_year,
                    dev_lag,
                    incurred_loss,
                    cumulative_paid_loss,
                    bulk_loss,
                    ye_1997_reserves
                FROM raw
            ),

            aggregated_loss AS (
                SELECT
                    acc_year,
                    dev_lag,
                    SUM(incurred_loss) AS incurred_loss,
                    SUM(cumulative_paid_loss) AS paid_loss,
                    SUM(bulk_loss) AS bulk_reserve,
                    SUM(ye_1997_reserves) AS ye_1997_reserves
                FROM unique_loss_rows
                GROUP BY acc_year, dev_lag
            ),

            join_ep AS (
                SELECT
                    aggregated_loss.acc_year,
                    aggregated_loss.dev_lag,
                    aggregated_loss.incurred_loss,
                    aggregated_loss.paid_loss,
                    aggregated_loss.bulk_reserve,                    
                    aggregated_ep.ep

                FROM aggregated_loss
                LEFT JOIN aggregated_ep
                    ON aggregated_loss.acc_year = aggregated_ep.acc_year

                ORDER BY aggregated_loss.acc_year, aggregated_loss.dev_lag
            ),

            add_case AS (
                select 
                    *,
                    incurred_loss - paid_loss - bulk_reserve AS case_reserve
                from join_ep
            ),

            add_reported as (
                select 
                    *,
                    paid_loss + case_reserve AS reported_loss

                from add_case
            )

            SELECT * FROM add_reported
        
                
    """).df()

    df.to_csv(DATA_DIR / "other_liability_data_clean.csv", index=False)
