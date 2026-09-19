import ast
import csv
import io
import random
import re
import string
import sys
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

# Default datetime range for DATETIME / TIMESTAMP columns
DATETIME_START: datetime = datetime(1999, 7, 10, 0, 0, 0)
DATETIME_END: datetime = datetime(2099, 12, 31, 23, 59, 59)

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
        self.nullable_p: Optional[float] = self._parse_nullable()  # NULLABLE(p)
        self.decimal_spec: Optional[Tuple[int, int]] = self._parse_decimal()  # DECIMAL(p,s)
        self.distribution: Optional[Dict[str, Any]] = self._parse_distribution()
        self.time_fsp: Optional[int] = self._parse_fsp()  # DATETIME(n) / TIMESTAMP(n)
        self.auto_start: Optional[int] = self._parse_auto_start()  # AUTO_INCREMENT START(n)
        self.ruler_id: Optional[str] = self._parse_ruler_id()  # RULERID("$@...")
        self.pair_fk: Optional[Tuple[str, str]] = self._parse_pair_fk()  # PAIRFK(col, pcol)

    def _parse_length(self) -> Optional[int]:
        """Parse VARCHAR/CHAR length from the attribute line."""
        m = re.search(r"(VARCHAR|CHAR)\s*\((\d+)\)", self.attrs, re.I)
        return int(m.group(2)) if m else None

    def _parse_skew(self) -> Optional[float]:
        """Parse SKEW(p) value from the attribute line."""
        m = re.search(r"SKEW\(([\d.]+)\)", self.attrs)
        return float(m.group(1)) if m else None

    def _parse_nullable(self) -> Optional[float]:
        """Parse NULLABLE(p) probability of a NULL value from the attribute line."""
        m = re.search(r"NULLABLE\(([\d.]+)\)", self.attrs)
        return float(m.group(1)) if m else None

    def _parse_fsp(self) -> Optional[int]:
        """Parse fractional-seconds precision from DATETIME(n) / TIMESTAMP(n)."""
        m = re.search(r"(DATETIME|TIMESTAMP)\s*\((\d+)\)", self.attrs, re.I)
        return int(m.group(2)) if m else None

    def _parse_auto_start(self) -> Optional[int]:
        """Parse AUTO_INCREMENT START(n) offset."""
        m = re.search(r"AUTO_INCREMENT\s+START\s*\((\d+)\)", self.attrs, re.I)
        return int(m.group(1)) if m else None

    def _parse_ruler_id(self) -> Optional[str]:
        """Parse RULERID(\"template\") whose '$' is filled with this row's PK value."""
        m = re.search(r'RULERID\("(.+?)"\)', self.attrs, re.I)
        return m.group(1) if m else None

    def _parse_pair_fk(self) -> Optional[Tuple[str, str]]:
        """Parse PAIRFK(fk_col, parent_col): reuse the parent row sampled by fk_col."""
        m = re.search(r"PAIRFK\(\s*`?(\w+)`?\s*,\s*`?(\w+)`?\s*\)", self.attrs, re.I)
        return (m.group(1), m.group(2)) if m else None

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


def rand_datetime(
    start: datetime = DATETIME_START,
    end: datetime = DATETIME_END,
    fsp: Optional[int] = None,
) -> str:
    """
    Generate a random datetime within [start, end], formatted for SQL/CSV.

    Args:
        start: Start datetime (inclusive).
        end: End datetime (inclusive).
        fsp: Fractional-seconds precision (0-6). When None or 0, seconds are
             integer; otherwise microseconds are truncated to fsp digits.

    Returns:
        A formatted datetime string (e.g. "2022-09-28 13:09:00" or
        "2022-09-28 13:09:00.123456").
    """
    delta_seconds = int((end - start).total_seconds())
    dt = start + timedelta(seconds=random.randint(0, max(delta_seconds, 0)))
    base = dt.strftime("%Y-%m-%d %H:%M:%S")
    if fsp and fsp > 0:
        frac = random.randint(0, 10**fsp - 1)
        base += f".{frac:0{fsp}d}"
    return base


def parse_timerange(attrs: str) -> Optional[Tuple[datetime, datetime]]:
    """
    Parse TIMERANGE("start", "end") from the attribute line.

    Args:
        attrs: Column definition line.

    Returns:
        A tuple (start, end) datetime, or None if not present.
    """
    m = re.search(r'TIMERANGE\(\s*"([^"]+)"\s*,\s*"([^"]+)"\s*\)', attrs, re.I)
    if not m:
        return None
    fmt = "%Y-%m-%d %H:%M:%S"
    return (
        datetime.strptime(m.group(1), fmt),
        datetime.strptime(m.group(2), fmt),
    )


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
    - RANGE(min,max) for integer / numeric types
    - TIMERANGE("start","end") for DATETIME/TIMESTAMP
    - DECIMAL(p,s)
    - SET(...)
    - RULER("...$...")
    - VARCHAR/CHAR/TEXT/BLOB random strings
    - DATE / DATETIME / TIMESTAMP random values
    - Integer family: TINYINT / SMALLINT / MEDIUMINT / INT / BIGINT

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

    # ---------- INTEGER FAMILY ----------
    int_bounds: Dict[str, Tuple[int, int]] = {
        "TINYINT": (0, 1),
        "SMALLINT": (-32768, 32767),
        "MEDIUMINT": (-8388608, 8388607),
        "INT": (-2147483648, 2147483647),
        "INTEGER": (-2147483648, 2147483647),
        "BIGINT": (-9223372036854775808, 9223372036854775807),
    }
    int_type = "INT" if col.type.startswith("INT") else col.type
    if int_type in int_bounds:
        type_lo, type_hi = int_bounds[int_type]
        if rng:
            lo = max(int(float(rng.group(1))), type_lo)
            hi = min(int(float(rng.group(2))), type_hi)
            return random.randint(lo, hi)
        if int_type == "TINYINT":
            return random.randint(0, 1)
        return random.randint(max(type_lo, 1), min(type_hi, 10000))

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

    # ---------- DATETIME / TIMESTAMP ----------
    if col.type in {"DATETIME", "TIMESTAMP"}:
        trange = parse_timerange(line)
        if trange:
            start_dt, end_dt = trange
        else:
            start_dt, end_dt = DATETIME_START, DATETIME_END
        return rand_datetime(start_dt, end_dt, col.time_fsp)

    # ---------- DATE ----------
    if col.type == "DATE":
        drange = parse_timerange(line)
        if drange:
            return rand_date(drange[0].date(), drange[1].date()).isoformat()
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

    def __init__(
        self,
        col: Column,
        weights: Dict[Any, float],
        n_rows: int,
        fallback_sampler: Optional[Any] = None,
    ) -> None:
        """
        Args:
            col: Column metadata (used to generate fallback values).
            weights: Mapping value -> ratio from the HISTOGRAM spec.
            n_rows: Total number of rows to generate.
            fallback_sampler: Optional zero-argument callable returning a
                              fallback value (e.g. parent FK sampler). When
                              None, values are generated from the column.

        Raises:
            ValueError: If the rounded quotas exceed n_rows (ratios sum > 1.0).
        """
        self.col = col
        self.fallback_sampler = fallback_sampler

        # Validate the ratio total independently of rounding. Independent
        # round() calls can overshoot by 1 even when ratios sum to exactly 1.0
        # (e.g. 0.7/0.3 with an odd n), so quotas are allocated with the
        # largest-remainder method rather than per-key rounding.
        sum_ratios = float(sum(weights.values()))
        if sum_ratios > 1.0 + 1e-9:
            raise ValueError(f"HISTOGRAM ratios exceed 1.0 on column {col.name}")

        exact_counts = [(value, ratio * n_rows) for value, ratio in weights.items()]
        floors = {value: int(exact) for value, exact in exact_counts}
        fractions = {value: exact - floors[value] for value, exact in exact_counts}

        # Number of explicit (non-fallback) slots after rounding the ratio sum.
        quota_target = min(n_rows, int(sum_ratios * n_rows + 0.5))
        remainder_slots = quota_target - sum(floors.values())

        # Hand the remaining slots to keys with the largest fractional parts.
        order = sorted(
            weights.keys(),
            key=lambda v: (-fractions[v], list(weights.keys()).index(v)),
        )
        allocated = dict(floors)
        for value in order:
            if remainder_slots <= 0:
                break
            allocated[value] += 1
            remainder_slots -= 1

        self.quotas: Dict[Any, int] = {
            value: count for value, count in allocated.items() if count > 0
        }

        total = sum(self.quotas.values())
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

        def non_key_value() -> Any:
            if self.fallback_sampler is not None:
                return self.fallback_sampler()
            return generate_base_value(self.col)

        value = non_key_value()
        attempts = 0
        while value in self.keys:
            attempts += 1
            if attempts > 1000:
                raise ValueError(
                    f"HISTOGRAM keys on column {self.col.name} cover the whole "
                    "value domain; make the ratios sum to 1.0 or reduce the keys."
                )
            value = non_key_value()
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


def sample_parent_pk(
    fk_ctx: Dict[str, Any],
    parent_table: str,
    parent_pk_col: str,
) -> Any:
    """
    Sample a parent primary-key value from the parent's FULL key domain.

    Unlike the bounded row reservoir, this covers every parent PK, so large
    child tables can draw far more distinct foreign keys than RESERVOIR_SIZE.

    Args:
        fk_ctx: Parent context built by generate_csv (keys: domains, pinned).
        parent_table: Parent table name.
        parent_pk_col: Parent's single primary-key column name.

    Returns:
        A parent PK value (native type).

    Raises:
        KeyError: If the parent has no stored single-column PK domain.
    """
    domain = fk_ctx["domains"][(parent_table, parent_pk_col)]
    return random.choice(domain)


def _rebuild_rulerid(template: str, pk_value: Any) -> str:
    """Fill a deterministic RULERID template with the given PK value."""
    return template.replace("$", str(pk_value))


def resolve_paired_value(
    fk_ctx: Dict[str, Any],
    parent_table: str,
    parent_pk_col: str,
    fk_value: Any,
    parent_col: str,
) -> Any:
    """
    Resolve a parent-column value paired with an already-chosen FK value.

    Lookup order:
      1. Pinned complete rows (hot FK keys pinned from child histograms).
      2. Lazily-built reservoir index (random non-hot keys that happen to be
         inside the reservoir).
      3. Deterministic rebuild: if the parent column is a single RULERID with
         no NULLABLE, the value is a pure function of the parent PK.
      4. Otherwise raise (the pairing cannot be resolved exactly).

    Args:
        fk_ctx: Parent context; carries pinned rows and a mutable cache dict.
        parent_table: Parent table name.
        parent_pk_col: Parent primary-key column referenced by the sibling FK.
        fk_value: The parent PK value this row already drew.
        parent_col: Parent column whose value is required.

    Returns:
        The paired parent-column value.
    """
    pinned = fk_ctx["pinned"].get(parent_table, {})
    if fk_value in pinned:
        return pinned[fk_value][parent_col]

    # Asking for the parent's own PK column: value is the key itself.
    if parent_col == parent_pk_col:
        return fk_value

    reservoir_rows = fk_ctx["reservoirs"].get(parent_table, [])
    cache = fk_ctx.setdefault("res_index", {})
    idx_key = (parent_table, parent_pk_col)
    if idx_key not in cache:
        cache[idx_key] = {r[parent_pk_col]: r for r in reservoir_rows}
    row = cache[idx_key].get(fk_value)
    if row is not None:
        return row[parent_col]

    # Deterministic rebuild from the parent column's RULERID template.
    parent_table_meta = fk_ctx["table_meta"].get(parent_table)
    if parent_table_meta is not None and parent_col in parent_table_meta.columns:
        pcol = parent_table_meta.columns[parent_col]
        if (
            pcol.ruler_id is not None
            and pcol.nullable_p is None
        ):
            return _rebuild_rulerid(pcol.ruler_id, fk_value)

    raise KeyError(
        f"Cannot pair parent column {parent_table}.{parent_col} for key "
        f"{fk_value!r}; it is outside the reservoir and is not a deterministic "
        "RULERID. Pin the key or make the column deterministic."
    )


def iter_table_rows(
    table: Table,
    n_rows: int,
    fk_samples: Dict[str, List[Dict[str, Any]]],
    skew_hots: Dict[str, Any],
    fk_ctx: Optional[Dict[str, Any]] = None,
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

    # Backwards-compatible default context when called without one (e.g. direct
    # unit tests): fall back to reservoir-only sampling.
    if fk_ctx is None:
        fk_ctx = {
            "domains": {},
            "pinned": {},
            "reservoirs": fk_samples,
            "res_index": {},
            "pk_cols": {},
            "table_meta": {},
        }

    for col_name, col in table.columns.items():
        is_single_pk = col_name in table.primary_key and len(table.primary_key) == 1

        dist = col.distribution
        if dist and not is_single_pk:
            if dist["type"] == "histogram":
                # For a histogram on a foreign-key column, the non-hot remainder
                # must still sample valid parent values. Prefer the parent's FULL
                # key domain; fall back to the bounded reservoir if no domain is
                # registered (standalone calls).
                fk_info = next(
                    (fk for fk in table.foreign_keys if fk[0] == col_name), None
                )
                fb_sampler = None
                if fk_info:
                    _, parent_table, parent_col = fk_info
                    domain_key = (parent_table, parent_col)
                    if domain_key in fk_ctx["domains"]:
                        fb_sampler = (
                            lambda dk=domain_key: sample_parent_pk(fk_ctx, dk[0], dk[1])
                        )
                    else:
                        parent_rows = fk_samples[parent_table]
                        fb_sampler = (
                            lambda pr=parent_rows, pc=parent_col: random.choice(pr)[pc]
                        )
                hist_plans[col_name] = HistogramPlan(
                    col, dist["weights"], n_rows, fb_sampler
                )
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
    # Per-column AUTO_INCREMENT counters, optionally offset by START(n).
    auto_inc: Dict[str, int] = {}
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
                if col_name not in auto_inc:
                    auto_inc[col_name] = col.auto_start or 0
                row[col_name] = auto_inc[col_name]
                auto_inc[col_name] += 1
                continue

            # ---------- NULLABLE(p) ----------
            # Evaluated before consuming any distribution/skew plan slot, so a
            # NULL row neither draws a plan value nor distorts the exact quotas.
            if col.nullable_p is not None and random.random() < col.nullable_p:
                row[col_name] = ""
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

            # ---------- PAIRFK ----------
            # Reuse the parent row already sampled by the sibling FK column so
            # paired fields (e.g. repo_id/repo_name) describe the same parent.
            if col.pair_fk is not None:
                fk_col, parent_col = col.pair_fk
                sibling_fk = next(
                    fk for fk in table.foreign_keys if fk[0] == fk_col
                )
                _, parent_table, sibling_parent_col = sibling_fk
                row[col_name] = resolve_paired_value(
                    fk_ctx,
                    parent_table,
                    sibling_parent_col,
                    row[fk_col],
                    parent_col,
                )
                continue

            # ---------- FOREIGN KEY ----------
            fk_info = next((fk for fk in table.foreign_keys if fk[0] == col_name), None)
            if fk_info:
                _, parent_table, parent_col = fk_info
                domain_key = (parent_table, parent_col)
                if domain_key in fk_ctx["domains"]:
                    row[col_name] = sample_parent_pk(
                        fk_ctx, parent_table, parent_col
                    )
                else:
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

        # ---------- RULERID: fill templates using this row's PK value ----------
        # Done after PK validation; for a single-column PK we substitute that
        # value, so fields like email become "user<id>@example.test" and match
        # the login workload predicates.
        if len(table.primary_key) == 1:
            pk_value = str(row[table.primary_key[0]])
            for col_name, col in table.columns.items():
                if col.ruler_id is not None:
                    row[col_name] = col.ruler_id.replace("$", pk_value)

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
    fk_ctx: Optional[Dict[str, Any]] = None,
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
        fk_ctx: Parent context (full domains / pinned rows); shared with the
                real pass so sampled values have representative sizes.

    Returns:
        Estimated row count (>= 1).
    """
    sample_rows = list(
        iter_table_rows(table, SAMPLE_ROWS, fk_samples, skew_hots, fk_ctx)
    )

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


def collect_pinned_keys(
    tables: Dict[str, "Table"],
) -> Dict[Tuple[str, str], Set[Any]]:
    """
    Pre-scan every table for FK HISTOGRAM hot keys that must stay resolvable.

    Child workloads reference concrete hot parent keys (e.g. repo 41986369 or
    actor 800000001). When generating the parent we pin the COMPLETE rows for
    those keys so paired columns (PAIRFK) can be resolved even though the keys
    are almost surely outside a bounded random reservoir.

    Args:
        tables: All parsed tables (schema-wide scan; child tables may appear
                later in SQL order).

    Returns:
        Mapping (parent_table, parent_pk_col) -> set of pinned key values.
    """
    pinned: Dict[Tuple[str, str], Set[Any]] = {}
    for child in tables.values():
        for child_col, parent_table, parent_col in child.foreign_keys:
            col = child.columns[child_col]
            if col.distribution and col.distribution.get("type") == "histogram":
                for key in col.distribution["weights"].keys():
                    pinned.setdefault((parent_table, parent_col), set()).add(key)
    return pinned


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

    # Pre-scan all children so that parent generation can pin hot FK keys.
    pinned_keys = collect_pinned_keys(tables)

    # State handed down to every iter_table_rows call.
    fk_samples: Dict[str, List[Dict[str, Any]]] = {}
    pk_domains: Dict[Tuple[str, str], List[Any]] = {}
    pinned_rows: Dict[str, Dict[Any, Dict[str, Any]]] = {}
    pk_col_of: Dict[str, str] = {}

    fk_ctx: Dict[str, Any] = {
        "domains": pk_domains,
        "pinned": pinned_rows,
        "reservoirs": fk_samples,
        "res_index": {},
        "pk_cols": pk_col_of,
        "table_meta": tables,
    }

    for name, table in tables.items():
        # Hot values are sampled once and shared between the estimation pass
        # and the generation pass (byte-size mode).
        skew_hots = build_skew_hots(table, fk_samples)

        # ---------- Resolve target row count ----------
        if table.byte_target is not None:
            n_rows = estimate_row_count(table, fk_samples, skew_hots, fk_ctx)
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

        # Track this table's single-column PK so its full domain can be exposed
        # to child tables. A single auto-increment PK yields a dense range and
        # is represented compactly as a NumPy array; otherwise we collect the
        # PK of every emitted row.
        single_pk = table.primary_key[0] if len(table.primary_key) == 1 else None
        pk_is_autoincrement = (
            single_pk is not None
            and "AUTO_INCREMENT" in table.columns[single_pk].attrs
        )
        # Dense auto-increment start (defaults to 0 like the generator counter).
        pk_start: Optional[int] = None
        if pk_is_autoincrement:
            pk_start = table.columns[single_pk].auto_start
            if pk_start is None:
                pk_start = 0

        # Keys of this table that children require to be resolvable.
        pin_targets = {
            key
            for (ptable, pcol), keys in pinned_keys.items()
            if ptable == name and (single_pk is None or pcol == single_pk)
            for key in keys
        }
        own_pinned: Dict[Any, Dict[str, Any]] = {}

        with open(output_dir / f"{name}.csv", "w", newline="", encoding="utf8") as f:
            writer = csv.DictWriter(f, fieldnames=list(table.columns.keys()))
            writer.writeheader()

            for row in iter_table_rows(
                table, n_rows, fk_samples, skew_hots, fk_ctx
            ):
                writer.writerow(row)
                count += 1

                if single_pk is not None and row[single_pk] in pin_targets:
                    # Keep an independent copy (row dict is reused by reference).
                    own_pinned[row[single_pk]] = dict(row)

                # Reservoir sampling so child tables can sample random complete
                # parent rows without keeping the whole table in memory.
                if len(reservoir) < RESERVOIR_SIZE:
                    reservoir.append(row)
                else:
                    j = random.randint(0, count - 1)
                    if j < RESERVOIR_SIZE:
                        reservoir[j] = row

        fk_samples[name] = reservoir
        pinned_rows[name] = own_pinned

        # Expose the parent's FULL PK domain to child FK sampling.
        if single_pk is not None:
            pk_col_of[name] = single_pk
            if pk_is_autoincrement:
                # Dense auto-increment range [start, start + count): compact and
                # exact, covering every PK (not just the reservoir).
                pk_domains[(name, single_pk)] = np.arange(
                    pk_start, pk_start + count, dtype=np.int64
                )
            elif count <= RESERVOIR_SIZE:
                pk_domains[(name, single_pk)] = [
                    r[single_pk] for r in fk_samples[name]
                ]
            else:
                # Non auto-inc PKs larger than the reservoir: read every PK back
                # from the just-written file to build the exact full domain.
                domain_vals: List[Any] = []
                with open(
                    output_dir / f"{name}.csv", "r", encoding="utf8"
                ) as rf:
                    rdr = csv.DictReader(rf)
                    for r in rdr:
                        domain_vals.append(r[single_pk])
                pk_domains[(name, single_pk)] = domain_vals

        actual_bytes = (output_dir / f"{name}.csv").stat().st_size
        message = f"[Done] {name}.csv generated ({count} rows, {human_size(actual_bytes)})"
        if table.byte_target is not None:
            message += f" [target {human_size(table.byte_target)}]"
        print(message)

    print("Have a nice day!")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generator.py <input_sql_file> [output_dir]")
        sys.exit(1)
    input_file = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else None
    generate_csv(input_file, out_dir)
