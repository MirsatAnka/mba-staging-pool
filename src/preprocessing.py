"""Transaction preprocessing, invoice-level sampling, and basket matrix construction.

Core academic and domain rules for Topic 6 (UCI Online Retail II):
1. Standardize schema across both Excel sheets ('Year 2009-2010' and 'Year 2010-2011').
2. Remove cancelled invoices (InvoiceNo starting with 'C') and invalid quantities/prices.
3. Handle missing Description and CustomerID entries.
4. Sample strictly by unique InvoiceNo with random_state=5018 (NEVER sample random rows).
5. Build a boolean transaction-item matrix (each row = 1 invoice/basket, each column = 1 item).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from mlxtend.preprocessing import TransactionEncoder

from src import GROUP_SEED

LOGGER = logging.getLogger(__name__)

# The UCI Excel sheets use slightly different header names ('Invoice', 'Price', 'Customer ID')
# than some CSV mirrors ('InvoiceNo', 'UnitPrice', 'CustomerID'). We map them to one schema.
COLUMN_ALIASES: dict[str, str] = {
    "Invoice": "InvoiceNo",
    "Price": "UnitPrice",
    "Customer ID": "CustomerID",
}


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names so Excel and CSV inputs share identical column labels."""
    # Step 1: Strip any accidental leading/trailing whitespace from column headers
    renamed_whitespace: dict[str, str] = {}
    for col in df.columns:
        renamed_whitespace[col] = str(col).strip()
    standardized_df = df.rename(columns=renamed_whitespace)

    # Step 2: Rename UCI Excel headers to our canonical schema if needed
    alias_mapping: dict[str, str] = {}
    for old_name, new_name in COLUMN_ALIASES.items():
        if old_name in standardized_df.columns and new_name not in standardized_df.columns:
            alias_mapping[old_name] = new_name

    if alias_mapping:
        standardized_df = standardized_df.rename(columns=alias_mapping)

    # Step 3: Verify that all required columns are present before proceeding
    required_columns = ("InvoiceNo", "Description", "Quantity", "UnitPrice")
    missing_columns: list[str] = []
    for req_col in required_columns:
        if req_col not in standardized_df.columns:
            missing_columns.append(req_col)

    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {missing_columns}. "
            f"Available columns: {list(standardized_df.columns)}"
        )

    return standardized_df


def load_raw_transactions(filepath: str | Path) -> pd.DataFrame:
    """Load raw transactions from an Excel workbook (all sheets) or a CSV file."""
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Transaction file not found: {path}")

    suffix = path.suffix.lower()
    LOGGER.info("Loading raw transactions from %s ...", path)

    # Read Invoice/StockCode columns as strings so prefixes like 'C' or leading zeros are kept intact
    string_dtypes = {
        "Invoice": str,
        "InvoiceNo": str,
        "StockCode": str,
    }

    if suffix in (".xlsx", ".xls"):
        # Online Retail II Excel workbook contains two sheets: 'Year 2009-2010' and 'Year 2010-2011'
        all_sheets = pd.read_excel(path, sheet_name=None, dtype=string_dtypes)
        sheet_frames: list[pd.DataFrame] = []

        for sheet_name, sheet_df in all_sheets.items():
            LOGGER.info("Loaded Excel sheet %r with %d rows.", sheet_name, len(sheet_df))
            standardized_sheet = standardize_columns(sheet_df)
            sheet_frames.append(standardized_sheet)

        raw_df = pd.concat(sheet_frames, ignore_index=True)

    elif suffix == ".csv":
        csv_df = pd.read_csv(path, dtype=string_dtypes, low_memory=False)
        raw_df = standardize_columns(csv_df)

    else:
        raise ValueError(f"Unsupported file extension {suffix!r} for {path}.")

    LOGGER.info("Total raw transaction rows loaded: %d", len(raw_df))
    return raw_df


def clean_transactions(
    df: pd.DataFrame,
    drop_missing_customer_id: bool = True,
) -> pd.DataFrame:
    """Remove cancelled invoices, invalid quantities/prices, and missing descriptions."""
    cleaned_df = standardize_columns(df).copy()
    initial_rows = len(cleaned_df)

    # Step 1: Drop rows where InvoiceNo or product Description is missing
    cleaned_df = cleaned_df.dropna(subset=["InvoiceNo", "Description"])

    # Optionally drop rows without a CustomerID (standard practice for Online Retail II)
    if drop_missing_customer_id and "CustomerID" in cleaned_df.columns:
        cleaned_df = cleaned_df.dropna(subset=["CustomerID"])

    # Step 2: Convert InvoiceNo to clean strings and remove cancelled/adjustment invoices
    # In Online Retail II, cancelled invoices start with 'C' (e.g. 'C536379')
    # and accounting debt adjustments start with 'A'.
    cleaned_df["InvoiceNo"] = cleaned_df["InvoiceNo"].astype(str).str.strip()
    upper_invoice = cleaned_df["InvoiceNo"].str.upper()

    is_cancelled = upper_invoice.str.startswith("C")
    is_adjustment = upper_invoice.str.startswith("A")
    is_blank_invoice = (cleaned_df["InvoiceNo"] == "") | (upper_invoice == "NAN")

    valid_invoice_mask = (~is_cancelled) & (~is_adjustment) & (~is_blank_invoice)
    cleaned_df = cleaned_df.loc[valid_invoice_mask].copy()

    # Step 3: Convert Quantity and UnitPrice to numbers and keep only strictly positive values
    cleaned_df["Quantity"] = pd.to_numeric(cleaned_df["Quantity"], errors="coerce")
    cleaned_df["UnitPrice"] = pd.to_numeric(cleaned_df["UnitPrice"], errors="coerce")
    cleaned_df = cleaned_df.dropna(subset=["Quantity", "UnitPrice"])

    positive_value_mask = (cleaned_df["Quantity"] > 0) & (cleaned_df["UnitPrice"] > 0)
    cleaned_df = cleaned_df.loc[positive_value_mask].copy()

    # Step 4: Normalize product Description text (uppercase, single spaces) and drop blanks
    cleaned_df["Description"] = (
        cleaned_df["Description"]
        .astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .str.upper()
    )
    invalid_descriptions = ("", "NAN", "NONE", "?")
    valid_desc_mask = ~cleaned_df["Description"].isin(invalid_descriptions)
    cleaned_df = cleaned_df.loc[valid_desc_mask].reset_index(drop=True)

    LOGGER.info(
        "Cleaned dataset: %d -> %d rows (%d unique invoices, %d unique products).",
        initial_rows,
        len(cleaned_df),
        cleaned_df["InvoiceNo"].nunique(),
        cleaned_df["Description"].nunique(),
    )
    return cleaned_df


def sample_invoices(
    df: pd.DataFrame,
    seed: int = GROUP_SEED,
    n_invoices: int = 5000,
) -> pd.DataFrame:
    """Sample unique invoices deterministically while keeping every item in each basket.

    IMPORTANT DOMAIN RULE:
    We NEVER sample individual rows because random row sampling would split
    an invoice in half and drop items from a customer's basket. Instead, we
    sample unique InvoiceNo values using random_state=5018 and keep all rows
    belonging to those selected invoices.
    """
    if n_invoices <= 0:
        raise ValueError(f"n_invoices must be a positive integer, got {n_invoices}.")

    if df.empty:
        raise ValueError("Cannot sample invoices from an empty DataFrame.")

    # Step 1: Extract and sort all unique InvoiceNo identifiers for 100% deterministic order
    unique_invoice_list = sorted(df["InvoiceNo"].unique().tolist())
    unique_invoice_series = pd.Series(unique_invoice_list)
    total_unique = len(unique_invoice_series)

    # Step 2: Determine actual sample size (cannot exceed available unique invoices)
    sample_size = min(n_invoices, total_unique)
    if sample_size < n_invoices:
        LOGGER.warning(
            "Requested %d invoices, but only %d valid unique invoices exist. Using all %d.",
            n_invoices,
            total_unique,
            sample_size,
        )

    # Step 3: Sample unique invoice IDs using the fixed group random_state (5018)
    sampled_invoice_series = unique_invoice_series.sample(
        n=sample_size,
        random_state=seed,
        replace=False,
    )
    sampled_invoice_ids = set(sampled_invoice_series.tolist())

    # Step 4: Keep ALL transaction rows belonging to the sampled invoices
    sampled_df = df.loc[df["InvoiceNo"].isin(sampled_invoice_ids)].copy()
    sampled_df = sampled_df.reset_index(drop=True)

    LOGGER.info(
        "Sampled %d unique invoices (random_state=%d) -> %d transaction rows, %d unique items.",
        sampled_df["InvoiceNo"].nunique(),
        seed,
        len(sampled_df),
        sampled_df["Description"].nunique(),
    )
    return sampled_df


def load_and_sample_data(
    filepath: str | Path,
    seed: int = GROUP_SEED,
    n_invoices: int = 5000,
    drop_missing_customer_id: bool = True,
) -> pd.DataFrame:
    """Load raw transactions, remove cancellations/invalid rows, and sample unique invoices."""
    # Step 1: Load raw transactions from disk
    raw_df = load_raw_transactions(filepath)

    # Step 2: Remove cancelled invoices ('C'), negative quantities/prices, and missing values
    cleaned_df = clean_transactions(
        raw_df,
        drop_missing_customer_id=drop_missing_customer_id,
    )

    # Step 3: Sample unique invoices deterministically (random_state=5018)
    sampled_df = sample_invoices(
        cleaned_df,
        seed=seed,
        n_invoices=n_invoices,
    )
    return sampled_df


def create_basket_matrix(
    df: pd.DataFrame,
    invoice_col: str = "InvoiceNo",
    item_col: str = "Description",
    sparse: bool = True,
) -> pd.DataFrame:
    """Transform transaction records into a boolean invoice-by-item matrix for mlxtend.

    Each row represents one unique invoice (market basket).
    Each column represents one unique product description.
    Cell value is True if the product appeared in that invoice (regardless of quantity),
    and False otherwise.
    """
    if df.empty:
        raise ValueError("Cannot build a basket matrix from an empty DataFrame.")

    # Step 1: Build a list of baskets (one list of unique product names per invoice)
    invoice_ids: list[str] = []
    transactions: list[list[str]] = []

    for invoice_no, group_df in df.groupby(invoice_col, sort=True):
        # Convert item names to unique sorted strings for this single invoice
        unique_items_in_basket = sorted(group_df[item_col].astype(str).unique().tolist())
        invoice_ids.append(str(invoice_no))
        transactions.append(unique_items_in_basket)

    # Step 2: Fit mlxtend's TransactionEncoder to map every unique item to a boolean column
    encoder = TransactionEncoder()
    encoded_matrix = encoder.fit(transactions).transform(transactions, sparse=sparse)

    # Step 3: Wrap the encoded matrix in a pandas DataFrame indexed by InvoiceNo
    if sparse:
        basket_matrix = pd.DataFrame.sparse.from_spmatrix(
            encoded_matrix,
            index=pd.Index(invoice_ids, name=invoice_col),
            columns=encoder.columns_,
        )
        # Explicitly set SparseDtype(bool, fill_value=False) so mlxtend treats it as boolean
        sparse_bool_type = pd.SparseDtype(bool, fill_value=False)
        basket_matrix = basket_matrix.astype(sparse_bool_type)
        matrix_density = float(basket_matrix.sparse.density)
    else:
        basket_matrix = pd.DataFrame(
            encoded_matrix,
            index=pd.Index(invoice_ids, name=invoice_col),
            columns=encoder.columns_,
            dtype=bool,
        )
        matrix_density = float(basket_matrix.to_numpy().mean())

    LOGGER.info(
        "Built %s boolean basket matrix of shape %s (density=%.4f%%).",
        "sparse" if sparse else "dense",
        basket_matrix.shape,
        matrix_density * 100.0,
    )
    return basket_matrix
