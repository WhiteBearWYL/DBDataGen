import ast
import csv
import io
import random
import re
import string
from collections import OrderedDict
from datetime import date, datetime, timedelta
from decimal import Decimal, ROUND_DOWN, getcontext
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple, Union

import numpy as np

CURRENT_TIME: str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Default date range for DATE columns
START_DATE: date = date(1999, 7, 10)
END_DATE: date = date(2099, 12, 31)

# Default string length for STRING columns
STRING_MIN_LENGTH: int = 1
STRING_MAX_LENGTH: int = 100

# Byte units supported in "SIZE = <n><unit>"
SIZE_UNITS: Dict[str, int] = {
    "B": 1,
    "KB": 1024,
    "MB": 1024**2,
    "GB": 1024**3,
    "TB": 1024**4,
}

# Number of rows sampled to estimate the average row size when SIZE is a byte target
SAMPLE_ROWS: int = 2000

# Maximum number of rows kept in memory per table for foreign-key sampling
RESERVOIR_SIZE: int = 100_000

# Chunk size for lazily sampling NORMAL / POISSON distributions
DISTRIBUTION_CHUNK: int = 8192


class Column:
    """
    Column metadata parsed from a CREATE TABLE statement line.

    This class extracts and stores column constraints / extensions embedded in the
    column definition line, such as:
    - VARCHAR/CHAR length
    - SKEW(p)
    - DECIMAL(p,s)
    - Distribution spec: DISTRI(NORMAL(mu,sigma)), DISTRI(POISSON(lam)), HISTOGRAM({...})
    """

    def __init__(self, name: str, col_type: str, attrs: str) -> None:
        """
        Initialize a Column.

        Args:
            name: Column name.
            col_type: SQL column type (e.g., INT, VARCHAR, DECIMAL).
            attrs: The full column definition line containing constraints and extensions
                   (e.g., AUTO_INCREMENT, RANGE, SET, RULER, SKEW, DISTRI, HISTOGRAM).
        """
        self.name: str = name
        self.type: str = col_type.upper()
        self.attrs: str = attrs  # AUTO_INCREMENT, SKEW, RANGE, SET, RULER, NOT NULL, ...
        self.max_length: Optional[int] = self._parse_length()
        self.skew_p: Optional[float] = self._parse_skew()  # SKEW(p)
        self.decimal_spec: Optional[Tuple[int, int]] = self._parse_decimal()  # DECIMAL(p,s)
        self.distribution: Optional[Dict[str, Any]] = self._parse_distribution()

    def _parse_length(self) -> Optional[int]:
        """Parse VARCHAR/CHAR length from the attribute line."""
        m = re.search(r"(VARCHAR|CHAR)\s*\((\d+)\)", self.attrs, re.I)
        return int(m.group(2)) if m else None

    def _parse_skew(self) -> Optional[float]:
        """Parse SKEW(p) value from the attribute line."""
        m = re.search(r"SKEW\(([\d.]+)\)", self.attrs)
        return float(m.group(1)) if m else None

    def _parse_decimal(self) -> Optional[Tuple[int, int]]:
        """
        Parse DECIMAL(p,s) specification.

        Returns:
            A tuple (p, s) where p is precision and s is scale, or None if not present.

        Raises:
            ValueError: If DECIMAL(p,s) is invalid (e.g., s >= p).
        """
        m = re.search(r"DECIMAL\s*\((\d+)\s*,\s*(\d+)\)", self.attrs, re.I)
        if m:
            p, s = int(m.group(1)), int(m.group(2))
            if s >= p:
                raise ValueError(f"Invalid DECIMAL({p},{s}) on column {self.name}")
            return p, s
        return None

    def _parse_distribution(self) -> Optional[Dict[str, Any]]:
        """
        Parse distribution specification.

        Supported:
        - HISTOGRAM({value: ratio, ...})
        - DISTRI(NORMAL(mu,sigma))
        - DISTRI(POISSON(lam))

        Returns:
            A dict describing the distribution, or None if not present.
        """
        # HISTOGRAM
        m = re.search(r"HISTOGRAM\s*\((\{.*?\})\)", self.attrs, re.I)
        if m:
            raw = ast.literal_eval(m.group(1))
            return {
                "type": "histogram",
                "weights": {k: float(v) for k, v in raw.items()},
            }

        # DISTRIBUTION
        m = re.search(r"DISTRI\s*\(\s*(\w+)\((.*?)\)\s*\)", self.attrs, re.I)
        if m:
            name = m.group(1).lower()
            params = [float(x) for x in m.group(2).split(",")]

            if name == "normal":
                return {"type": "normal", "mu": params[0], "sigma": params[1]}
            if name == "poisson":
                return {"type": "poisson", "lam": params[0]}

        return None


class Table:
    """
    Table metadata parsed from a CREATE TABLE block.

    Attributes:
        name: Table name.
        size: Number of rows to generate (SIZE = N), or None in byte-size mode.
        byte_target: Target output size in bytes (SIZE = 1GB), or None in row mode.
        columns: Ordered mapping of column name -> Column.
        primary_key: List of primary key column names.
        foreign_keys: List of foreign key tuples (col, ref_table, ref_col).
    """

    def __init__(
        self,
        name: str,
        size: Optional[int] = None,
        byte_target: Optional[int] = None,
    ) -> None:
        """
        Initialize a Table.

        Args:
            name: Table name.
            size: Number of rows to generate (row-count mode).
            byte_target: Target output size in bytes (data-volume mode).
        """
        self.name: str = name
        self.size: Optional[int] = size
        self.byte_target: Optional[int] = byte_target
        self.columns: "OrderedDict[str, Column]" = OrderedDict()
        self.primary_key: List[str] = []
        self.foreign_keys: List[Tuple[str, str, str]] = []  # (col, ref_table, ref_col)


def parse_size_spec(spec: str) -> Tuple[Optional[int], Optional[int]]:
    """
    Parse a SIZE specification.

    Supported forms:
    - "100"     -> row count
    - "1GB"     -> target data volume (units: B, KB, MB, GB, TB; decimals allowed)

    Args:
        spec: Raw SIZE value text from the DDL.

    Returns:
        A tuple (row_count, byte_target); exactly one of them is not None.

    Raises:
        ValueError: If the specification is invalid.
    """
    m = re.fullmatch(r"([\d.]+)\s*(B|KB|MB|GB|TB)?", spec.strip(), re.I)
    if m is None:
        raise ValueError(f"Invalid SIZE specification: {spec!r}")

    num = float(m.group(1))
    unit = (m.group(2) or "").upper()

    if not unit:
        if num <= 0 or not num.is_integer():
            raise ValueError(f"Row-count SIZE must be a positive integer: {spec!r}")
        return int(num), None

    return None, max(1, int(num * SIZE_UNITS[unit]))


def parse_sql(sql_text: str) -> "OrderedDict[str, Table]":
    """
    Parse SQL text containing one or more CREATE TABLE blocks with SIZE annotations.

    Expected DDL pattern:
        CREATE TABLE `table_name` (
            ...
        ) ... SIZE = <N>;            -- row count
        ) ... SIZE = <n><unit>;      -- target data volume, unit in B/KB/MB/GB/TB

    Args:
        sql_text: Full SQL file content.

    Returns:
        Ordered mapping: table_name -> Table object.
    """
    tables: "OrderedDict[str, Table]" = OrderedDict()
    blocks = re.findall(
        r"CREATE TABLE.*?SIZE\s*=\s*[\d.]+\s*(?:B|KB|MB|GB|TB)?\s*;",
        sql_text,
        re.S | re.I,
    )

    for block in blocks:
        table_name_match = re.search(r"CREATE TABLE\s+`(\w+)`", block)
        size_match = re.search(r"SIZE\s*=\s*([\d.]+)\s*(B|KB|MB|GB|TB)?", block, re.I)
        if table_name_match is None or size_match is None:
            continue

        table_name = table_name_match.group(1)
        size_spec = size_match.group(1) + (size_match.group(2) or "")
        size, byte_target = parse_size_spec(size_spec)
        table = Table(table_name, size=size, byte_target=byte_target)

        # Columns
        for line in block.splitlines():
            col_match = re.match(r"\s*`(\w+)`\s+(\w+)", line)
            if col_match:
                col, ctype = col_match.groups()
                table.columns[col] = Column(col, ctype.upper(), line)

        # Primary key
        pk = re.search(r"PRIMARY KEY\s*\((.*?)\)", block)
        if pk:
            table.primary_key = [x.strip(" `") for x in pk.group(1).split(",")]

        # Foreign keys
        for fk in re.finditer(
            r"FOREIGN KEY\s*\(`(\w+)`\)\s+REFERENCES\s+`(\w+)`\s*\(`(\w+)`\)",
            block,
        ):
            table.foreign_keys.append(fk.groups())  # type: ignore[arg-type]

        tables[table_name] = table

    return tables


def rand_string(min_len: int = STRING_MIN_LENGTH, max_len: int = STRING_MAX_LENGTH) -> str:
    """
    Generate a random ASCII letters string.

    Args:
        min_len: Minimum string length.
        max_len: Maximum string length.

    Returns:
        A random string. Returns an empty string if the length range is invalid.
    """
    if min_len < 0 or max_len < min_len or max_len == 0:
        return ""
    length = random.randint(min_len, max_len)
    return "".join(random.choices(string.ascii_letters, k=length))


def rand_date(start: date = START_DATE, end: date = END_DATE) -> date:
    """
    Generate a random date within [start, end].

    Args:
        start: Start date (inclusive).
        end: End date (inclusive).

    Returns:
        A random date.
    """
    delta = (end - start).days
    return start + timedelta(days=random.randint(0, delta))


def apply_ruler(rule: str, max_len: int) -> str:
    """
    Apply a RULER template to generate a string.

    The placeholder '$' will be replaced by a random string so that the resulting
    length does not exceed max_len.

    Example:
        rule = "$@email.com"

    Args:
        rule: RULER template string, where '$' indicates the variable part.
        max_len: Maximum total length of the produced string.

    Returns:
        The generated string after placeholder substitution.
    """
    fixed_len = len(rule.replace("$", ""))
    remain = max_len - fixed_len
    if remain <= 0:
        return rule.replace("$", "")
    return rule.replace("$", rand_string(max_len=remain))


def generate_decimal_value(
    precision: int,
    scale: int,
    min_value: Optional[Union[int, float]] = None,
    max_value: Optional[Union[int, float]] = None,
) -> Decimal:
    """
    Generate a Decimal value conforming to DECIMAL(precision, scale).

    Args:
        precision: Total number of digits (p).
        scale: Number of fractional digits (s).
        min_value: Optional lower bound for the integer part.
        max_value: Optional upper bound for the integer part.

    Returns:
        A quantized Decimal value with the requested scale.
    """
    int_digits = precision - scale
    max_int = 10**int_digits - 1

    lo = float(min_value) if min_value is not None else 0.0
    hi = float(max_value) if max_value is not None else float(max_int)
    if hi > max_int:
        hi = float(max_int)

    integer_part = random.randint(int(lo), int(hi))
    frac_part = random.randint(0, 10**scale - 1)

    value = Decimal(f"{integer_part}.{frac_part:0{scale}d}")
    getcontext().prec = precision
    return value.quantize(Decimal(f"1.{'0' * scale}"), rounding=ROUND_DOWN)


def generate_base_value(col: Column) -> Any:
    """
    Generate a base value for a column according to its constraints.

    Supported constraints / extensions:
    - RANGE(min,max) for numeric types (INT/FLOAT/DOUBLE/DECIMAL)
    - DECIMAL(p,s)
    - SET(...)
    - RULER("...$...")
    - VARCHAR/CHAR/TEXT/BLOB random strings
    - DATE random date (ISO format string)

    Args:
        col: Column metadata.

    Returns:
        A generated value. For DECIMAL, returns a Decimal (caller may convert to string).
    """
    line = col.attrs

    # ---------- RANGE ----------
    rng = re.search(r"RANGE\(([-\d.]+),\s*([-\d.]+)\)", line)

    # ---------- DECIMAL ----------
    if col.decimal_spec:
        precision, scale = col.decimal_spec
        min_v: Optional[float] = None
        max_v: Optional[float] = None
        if rng:
            min_v, max_v = float(rng.group(1)), float(rng.group(2))
        return generate_decimal_value(precision, scale, min_v, max_v)

    # ---------- FLOAT / DOUBLE ----------
    if col.type in {"FLOAT", "DOUBLE"}:
        min_v, max_v = (0.0, 1000.0)
        if rng:
            min_v, max_v = float(rng.group(1)), float(rng.group(2))
        val = random.uniform(min_v, max_v)
        return round(val, 6 if col.type == "FLOAT" else 10)

    # ---------- INT ----------
    if col.type.startswith("INT"):
        if rng:
            return random.randint(int(float(rng.group(1))), int(float(rng.group(2))))
        return random.randint(1, 10000)

    # ---------- SET ----------
    st = re.search(r"SET\((.*?)\)", line)
    if st:
        return random.choice([x.strip(" '") for x in st.group(1).split(",")])

    # ---------- RULER ----------
    ruler = re.search(r'RULER\("(.+?)"\)', line)
    if ruler:
        rule = ruler.group(1)
        fixed_len = len(rule.replace("$", ""))
        max_len = col.max_length if col.max_length is not None else fixed_len + 8
        remain = max_len - fixed_len
        if remain <= 0:
            return rule.replace("$", "")
        return rule.replace("$", rand_string(max_len=remain))

    # ---------- STRING ----------
    if col.type.startswith(("VARCHAR", "CHAR")):
        max_len = col.max_length if col.max_length is not None else 10
        return rand_string(max_len=max_len)
    if col.type.startswith(("TEXT", "BLOB")):
        return rand_string()

    # ---------- DATE ----------
    if col.type == "DATE":
        return rand_date().isoformat()

    return None


class HistogramPlan:
    """
    Streaming plan for a HISTOGRAM column.

    Allocates explicit values with exact per-value quotas (ratio * n_rows) and fills
    the remaining rows with generated values that do not overlap the histogram keys.
    Values are drawn sequentially so that the final counts match the quotas exactly
    while their positions remain uniformly shuffled.
    """

    def __init__(self, col: Column, weights: Dict[Any, float], n_rows: int) -> None:
        """
        Args:
            col: Column metadata (used to generate fallback values).
            weights: Mapping value -> ratio from the HISTOGRAM spec.
            n_rows: Total number of rows to generate.

        Raises:
            ValueError: If the rounded quotas exceed n_rows (ratios sum > 1.0).
        """
        self.col = col
        self.quotas: Dict[Any, int] = {}
        for value, ratio in weights.items():
            count = int(round(ratio * n_rows))
            if count > 0:
                self.quotas[value] = count

        total = sum(self.quotas.values())
        if total > n_rows:
            raise ValueError(f"HISTOGRAM ratios exceed 1.0 on column {col.name}")

        self.total_quota: int = total
        self.remaining_rows: int = n_rows
        self.keys: Set[Any] = set(weights.keys())

    def draw(self) -> Any:
        """Draw the value for the next generated row."""
        self.remaining_rows -= 1

        if self.total_quota > 0:
            # Sequential sampling: probability that this row takes one of the
            # remaining explicit quota slots.
            denom = max(self.remaining_rows + 1, 1)
            if random.random() < self.total_quota / denom:
                r = random.uniform(0.0, float(self.total_quota))
                acc = 0.0
                for value, count in self.quotas.items():
                    if count <= 0:
                        continue
                    acc += count
                    if r <= acc:
                        self.quotas[value] -= 1
                        self.total_quota -= 1
                        return value

        return self._fallback()

    def rollback(self, value: Any) -> None:
        """
        Undo a draw() whose row was rejected (e.g. PK collision), so that the
        final per-value counts stay exact.

        Args:
            value: The value previously returned by draw().
        """
        self.remaining_rows += 1
        if value in self.quotas:
            self.quotas[value] += 1
            self.total_quota += 1

    def _fallback(self) -> Any:
        """Generate a value that does not collide with explicit histogram keys."""
        value = generate_base_value(self.col)
        while value in self.keys:
            value = generate_base_value(self.col)
        return value


class LazySampler:
    """
    Chunked lazy sampler for NORMAL / POISSON distributions.

    Values are sampled in fixed-size chunks on demand so that arbitrarily large
    tables do not require materializing the whole sample array in memory.
    """

    def __init__(self, dist: Dict[str, Any], chunk_size: int = DISTRIBUTION_CHUNK) -> None:
        """
        Args:
            dist: Distribution spec dict ("normal" or "poisson").
            chunk_size: Number of values sampled per refill.
        """
        self.dist = dist
        self.chunk_size = chunk_size
        self.buffer: List[Any] = []

    def next(self) -> Any:
        """Return the next sampled value."""
        if not self.buffer:
            self._refill()
        return self.buffer.pop()

    def pushback(self, value: Any) -> None:
        """Return a consumed value (from a rejected row) to the buffer."""
        self.buffer.append(value)

    def _refill(self) -> None:
        if self.dist["type"] == "normal":
            arr = np.random.normal(self.dist["mu"], self.dist["sigma"], self.chunk_size)
        else:
            arr = np.random.poisson(self.dist["lam"], self.chunk_size)
        self.buffer = arr.tolist()
        self.buffer.reverse()  # pop() yields values in sampling order


class SkewPlan:
    """
    Streaming plan for a SKEW(p) column.

    Exactly k = round(p * n_rows) rows receive the hot value; the skewed positions
    are chosen sequentially so they are uniformly distributed over all rows.
    """

    def __init__(self, k: int, hot: Any, n_rows: int) -> None:
        """
        Args:
            k: Number of rows that must receive the hot value.
            hot: The repeated hot value.
            n_rows: Total number of rows to generate.
        """
        self.remaining_k: int = k
        self.hot: Any = hot
        self.remaining_rows: int = n_rows

    def draw(self) -> Optional[Any]:
        """Return the hot value if this row is skewed, otherwise None."""
        self.remaining_rows -= 1
        if self.remaining_k <= 0:
            return None
        denom = max(self.remaining_rows + 1, 1)
        if random.random() < self.remaining_k / denom:
            self.remaining_k -= 1
            return self.hot
        return None

    def rollback(self) -> None:
        """
        Undo a draw() that consumed a skew slot for a rejected row, so that the
        total skew count stays exact.
        """
        self.remaining_rows += 1
        self.remaining_k += 1


def postprocess_distribution_value(col: Column, raw: Any) -> Any:
    """
    Post-process a value drawn from a distribution plan (RANGE clipping,
    SET mapping, string truncation, DECIMAL serialization).

    Args:
        col: Column metadata.
        raw: Raw value drawn from the distribution plan.

    Returns:
        The CSV-ready value.
    """
    # RANGE clipping
    rng = re.search(r"RANGE\(([-\d.]+),\s*([-\d.]+)\)", col.attrs)
    if rng:
        lo, hi = float(rng.group(1)), float(rng.group(2))
        raw = min(max(raw, lo), hi)

    # SET mapping (by index)
    st = re.search(r"SET\((.*?)\)", col.attrs)
    if st:
        options = [x.strip(" '") for x in st.group(1).split(",")]
        raw = options[int(abs(raw)) % len(options)]

    # String truncation
    if col.type.startswith(("VARCHAR", "CHAR")) and col.max_length is not None:
        raw = str(raw)[: col.max_length]

    # DECIMAL is serialized as string for CSV
    if col.decimal_spec:
        raw = str(raw)

    return raw


def build_skew_hots(
    table: Table,
    fk_samples: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """
    Sample the hot values for all SKEW(p) columns of a table.

    Hot values are sampled once per table and shared between the row-count
    estimation pass and the actual generation pass (byte-size mode) so that
    the estimated average row size matches the real one.

    Args:
        table: Table metadata.
        fk_samples: Parent table row reservoirs for FK sampling.

    Returns:
        Mapping: col_name -> hot value.

    Raises:
        ValueError: If a skewed FK column references a table not generated yet.
    """
    hots: Dict[str, Any] = {}
    for col_name, col in table.columns.items():
        if col.skew_p is None or col_name in table.primary_key:
            continue

        # Choose the hot value: for FK columns, sample an existing parent value.
        fk_info = next((fk for fk in table.foreign_keys if fk[0] == col_name), None)
        if fk_info:
            _, parent_table, parent_col = fk_info
            parent_rows = fk_samples.get(parent_table) or []
            if not parent_rows:
                raise ValueError(
                    f"Skew FK column {table.name}.{col_name} requires parent table "
                    f"{parent_table} generated first."
                )
            hots[col_name] = random.choice(parent_rows)[parent_col]
        else:
            hots[col_name] = generate_base_value(col)

    return hots


def iter_table_rows(
    table: Table,
    n_rows: int,
    fk_samples: Dict[str, List[Dict[str, Any]]],
    skew_hots: Dict[str, Any],
) -> Iterator[Dict[str, Any]]:
    """
    Stream rows for a single table.

    Guarantees (best-effort with bounded retries):
    - Primary key / composite primary key uniqueness
    - Foreign key referential integrity (samples from parent row reservoirs)
    - Global plan semantics for HISTOGRAM / DISTRI / SKEW (exact counts)

    Rows are yielded one by one so that arbitrarily large tables can be written
    to CSV without holding them in memory.

    Args:
        table: Table metadata.
        n_rows: Number of rows to generate.
        fk_samples: Parent table name -> reservoir of parent rows for FK sampling.
        skew_hots: Hot values for SKEW columns (see build_skew_hots).

    Yields:
        Row dicts.

    Raises:
        RuntimeError: If unable to generate the required number of unique rows
                      within the attempt limit.
    """
    # ===================== Build streaming plans =====================
    hist_plans: Dict[str, HistogramPlan] = {}
    samplers: Dict[str, LazySampler] = {}
    skew_plans: Dict[str, SkewPlan] = {}

    for col_name, col in table.columns.items():
        is_single_pk = col_name in table.primary_key and len(table.primary_key) == 1

        dist = col.distribution
        if dist and not is_single_pk:
            if dist["type"] == "histogram":
                hist_plans[col_name] = HistogramPlan(col, dist["weights"], n_rows)
            elif dist["type"] in ("normal", "poisson"):
                samplers[col_name] = LazySampler(dist)

        p = col.skew_p
        if p is not None and col_name in skew_hots:
            k = int(round(p * n_rows))
            k = max(0, min(n_rows, k))
            if k > 0:
                skew_plans[col_name] = SkewPlan(k, skew_hots[col_name], n_rows)

    # ===================== Main loop =====================
    # A single auto-increment PK can never collide; skipping its tracking avoids
    # holding every PK value in memory for huge tables.
    pk_is_auto_inc = (
        len(table.primary_key) == 1
        and table.primary_key[0] in table.columns
        and "AUTO_INCREMENT" in table.columns[table.primary_key[0]].attrs
    )
    pk_seen: Set[Tuple[Any, ...]] = set()
    auto_inc = 0
    attempt = 0
    max_attempts = n_rows * 10  # Prevent infinite loops
    generated = 0

    while generated < n_rows and attempt < max_attempts:
        attempt += 1
        row: Dict[str, Any] = {}
        # Values consumed from streaming plans in this attempt, recorded so they
        # can be rolled back if the row is rejected by the PK check.
        row_hist_raw: Dict[str, Any] = {}
        row_sampler_raw: Dict[str, Any] = {}
        row_skewed: List[str] = []

        for col_name, col in table.columns.items():
            # ---------- AUTO_INCREMENT ----------
            if "AUTO_INCREMENT" in col.attrs:
                row[col_name] = auto_inc
                auto_inc += 1
                continue

            # ---------- HISTOGRAM ----------
            if col_name in hist_plans:
                raw = hist_plans[col_name].draw()
                row_hist_raw[col_name] = raw
                row[col_name] = postprocess_distribution_value(col, raw)
                continue

            # ---------- NORMAL / POISSON ----------
            if col_name in samplers:
                raw = samplers[col_name].next()
                row_sampler_raw[col_name] = raw
                row[col_name] = postprocess_distribution_value(col, raw)
                continue

            # ---------- SKEW ----------
            if col_name in skew_plans:
                hot = skew_plans[col_name].draw()
                if hot is not None:
                    row_skewed.append(col_name)
                    row[col_name] = hot
                    continue

            # ---------- FOREIGN KEY ----------
            fk_info = next((fk for fk in table.foreign_keys if fk[0] == col_name), None)
            if fk_info:
                _, parent_table, parent_col = fk_info
                parent_rows = fk_samples[parent_table]
                row[col_name] = random.choice(parent_rows)[parent_col]
                continue

            # ---------- BASE GENERATOR ----------
            value = generate_base_value(col)

            # DECIMAL is serialized as string for CSV
            if col.decimal_spec:
                value = str(value)

            row[col_name] = value

        # ===================== Primary key check =====================
        if table.primary_key and not pk_is_auto_inc:
            pk = tuple(row[k] for k in table.primary_key)
            if pk in pk_seen:
                # Roll back plan slots consumed by this rejected row.
                for col_name, raw in row_hist_raw.items():
                    hist_plans[col_name].rollback(raw)
                for col_name, raw in row_sampler_raw.items():
                    samplers[col_name].pushback(raw)
                for col_name in row_skewed:
                    skew_plans[col_name].rollback()
                continue
            pk_seen.add(pk)

        generated += 1
        yield row

    # ===================== Failure guard =====================
    if generated < n_rows:
        raise RuntimeError(
            f"Failed to generate {n_rows} unique rows for table {table.name}. "
            f"Generated {generated} rows. "
            f"Possible reasons: excessive skew, tight PK constraints."
        )


def estimate_row_count(
    table: Table,
    fk_samples: Dict[str, List[Dict[str, Any]]],
    skew_hots: Dict[str, Any],
) -> int:
    """
    Estimate the number of rows needed to reach table.byte_target.

    Generates SAMPLE_ROWS sample rows, serializes them as CSV in memory and
    derives the average row size.

    Args:
        table: Table metadata (byte_target must be set).
        fk_samples: Parent table row reservoirs for FK sampling.
        skew_hots: Hot values for SKEW columns, shared with the real generation
                   pass to keep the estimated row size representative.

    Returns:
        Estimated row count (>= 1).
    """
    sample_rows = list(iter_table_rows(table, SAMPLE_ROWS, fk_samples, skew_hots))

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(table.columns.keys()))
    writer.writeheader()
    writer.writerows(sample_rows)
    sample_bytes = len(buf.getvalue().encode("utf8"))

    avg_bytes = sample_bytes / max(len(sample_rows), 1)
    if avg_bytes <= 0:
        avg_bytes = 1.0

    assert table.byte_target is not None
    return max(1, int(table.byte_target / avg_bytes))


def human_size(num_bytes: int) -> str:
    """Format a byte count as a human-readable string (e.g. 1.0GB)."""
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{int(size)}{unit}" if unit == "B" else f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}TB"


def generate_csv(sql_file: Union[str, Path], out_dir: Optional[Union[str, Path]] = None) -> None:
    """
    Generate CSV files for all tables defined in the SQL file.

    Tables are generated in SQL order. Each table streams its rows directly to
    disk; per table only a bounded reservoir of rows is kept in memory for
    foreign-key sampling by child tables.

    Args:
        sql_file: Path to the SQL file containing CREATE TABLE blocks with SIZE annotations.
        out_dir: Output directory path. If None, a timestamped folder will be used.

    Returns:
        None
    """
    sql_path = Path(sql_file)
    sql_text = sql_path.read_text(encoding="utf8")

    tables = parse_sql(sql_text)

    output_dir = Path(out_dir) if out_dir is not None else Path(f"output_{CURRENT_TIME}")
    output_dir.mkdir(exist_ok=True)

    fk_samples: Dict[str, List[Dict[str, Any]]] = {}

    for name, table in tables.items():
        # Hot values are sampled once and shared between the estimation pass
        # and the generation pass (byte-size mode).
        skew_hots = build_skew_hots(table, fk_samples)

        # ---------- Resolve target row count ----------
        if table.byte_target is not None:
            n_rows = estimate_row_count(table, fk_samples, skew_hots)
            print(
                f"[Info] {name}: target size {human_size(table.byte_target)}, "
                f"estimated {n_rows} rows"
            )
        else:
            assert table.size is not None
            n_rows = table.size

        # ---------- Stream rows to CSV ----------
        reservoir: List[Dict[str, Any]] = []
        count = 0

        with open(output_dir / f"{name}.csv", "w", newline="", encoding="utf8") as f:
            writer = csv.DictWriter(f, fieldnames=list(table.columns.keys()))
            writer.writeheader()

            for row in iter_table_rows(table, n_rows, fk_samples, skew_hots):
                writer.writerow(row)
                count += 1

                # Reservoir sampling so child tables can sample FK values
                # without keeping the whole table in memory.
                if len(reservoir) < RESERVOIR_SIZE:
                    reservoir.append(row)
                else:
                    j = random.randint(0, count - 1)
                    if j < RESERVOIR_SIZE:
                        reservoir[j] = row

        fk_samples[name] = reservoir

        actual_bytes = (output_dir / f"{name}.csv").stat().st_size
        message = f"[Done] {name}.csv generated ({count} rows, {human_size(actual_bytes)})"
        if table.byte_target is not None:
            message += f" [target {human_size(table.byte_target)}]"
        print(message)

    print("Have a nice day!")


if __name__ == "__main__":
    generate_csv("create_table.sql")
