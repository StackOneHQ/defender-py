"""Tier-3 reviewer input: record-oriented serialization (ENG-2455) and chunking (ENG-1339).

A port of the TypeScript package's ``formatRecordsForTier3`` / ``chunkTier3Input``. The shared
conformance fixtures pin the output byte for byte, so this follows the TS code closely:

- Lengths and cuts are in UTF-16 code units, as in JS. Strings are serialized in a UTF-16 view
  (astral characters as surrogate pairs) and converted back when the chunks are returned.
- Object keys follow JS ``Object.entries`` order: integer-like keys first, ascending.
- Non-string scalars render like JS ``String(v)`` (``true``, ``null``, ``1e-7``, ``1e+21``).
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Callable, Mapping, Sequence
from collections.abc import Set as AbstractSet
from decimal import Decimal
from typing import Any

from ..config import MAX_TRAVERSAL_DEPTH

# Line breaks (incl. U+2028/U+2029/NEL): flattened in keys, split in values, so neither can
# forge a bare line or a `\n\n` record boundary.
_LINE_BREAKS = re.compile("[\r\n\u2028\u2029\u0085]")
_LINE_BREAKS_RUN = re.compile("[\r\n\u2028\u2029\u0085]+")
_ASTRAL = re.compile("[\U00010000-\U0010ffff]")
_SURROGATE = re.compile("[\ud800-\udfff]")
_INDEX_KEY = re.compile(r"(?:0|[1-9][0-9]*)")
_MAX_ARRAY_INDEX = 2**32 - 2

# A large array of only non-string scalars collapses to `key: [N …]`.
ARRAY_SUMMARY_THRESHOLD = 32
# Per record, non-string scalars use at most this fraction of the share (anti-crowd-out).
SCALAR_BUDGET_FRACTION = 0.5
# Per-item floor reserved for each not-yet-serialized sibling.
MIN_FIELD_CHARS = 128
CHUNK_OVERLAP_CHARS = 500
DEFAULT_MAX_CHUNKS = 5

_BINARY = (bytes, bytearray, memoryview)


# --- UTF-16 view -----------------------------------------------------------------------------


def _pair(m: re.Match[str]) -> str:
    c = ord(m[0]) - 0x10000
    return chr(0xD800 + (c >> 10)) + chr(0xDC00 + (c & 0x3FF))


def to_utf16(s: str) -> str:
    """``s`` with astral characters split into surrogate pairs, so ``len`` and slicing match JS."""
    return _ASTRAL.sub(_pair, s) if _ASTRAL.search(s) else s


def from_utf16(s: str) -> str:
    """Join surrogate pairs back into characters. An unpaired surrogate becomes U+FFFD, so a
    provider that encodes to UTF-8 can't be made to fail (and fail open) by one."""
    if not _SURROGATE.search(s):
        return s
    return s.encode("utf-16-le", "surrogatepass").decode("utf-16-le", "replace")


def _utf16_len(s: str) -> int:
    if not _ASTRAL.search(s):
        return len(s)
    return len(s.encode("utf-16-le", "surrogatepass")) // 2


def _is_high(c: str) -> bool:
    return "\ud800" <= c <= "\udbff"


def _is_low(c: str) -> bool:
    return "\udc00" <= c <= "\udfff"


def safe_end(s: str, end: int) -> int:
    """An end index <= ``end`` that doesn't split a surrogate pair."""
    if 0 < end < len(s) and _is_high(s[end - 1]) and _is_low(s[end]):
        return end - 1
    return end


def safe_start(s: str, start: int) -> int:
    """A start index >= ``start`` that doesn't split a surrogate pair."""
    if 0 < start < len(s) and _is_low(s[start]) and _is_high(s[start - 1]):
        return start + 1
    return start


# --- JS value semantics ----------------------------------------------------------------------


def _js_number(x: int | float) -> str:
    """JS ``String(n)``. Python ints below 1e21 print exactly (JS would round past 2**53)."""
    if isinstance(x, int):
        if abs(x) < 10**21:
            return str(x)
        try:
            x = float(x)
        except OverflowError:
            return "Infinity" if x > 0 else "-Infinity"
    if x != x:
        return "NaN"
    if x in (float("inf"), float("-inf")):
        return "Infinity" if x > 0 else "-Infinity"
    if x == 0:
        return "0"
    sign = "-" if x < 0 else ""
    _, digit_tuple, exp = Decimal(repr(abs(x))).as_tuple()
    digits = "".join(map(str, digit_tuple)).rstrip("0") or "0"
    exp = int(exp) + (len(digit_tuple) - len(digits))
    k = len(digits)
    n = exp + k  # decimal point position
    if k <= n <= 21:
        out = digits + "0" * (n - k)
    elif 0 < n <= 21:
        out = f"{digits[:n]}.{digits[n:]}"
    elif -6 < n <= 0:
        out = "0." + "0" * (-n) + digits
    else:
        e = n - 1
        mant = digits if k == 1 else f"{digits[0]}.{digits[1:]}"
        out = f"{mant}e{'+' if e >= 0 else '-'}{abs(e)}"
    return sign + out


def _js_string(v: Any) -> str:
    if v is True:
        return "true"
    if v is False:
        return "false"
    if isinstance(v, (int, float)):
        return _js_number(v)
    return str(v)


def _is_non_string_scalar(v: Any) -> bool:
    return v is None or isinstance(v, (bool, int, float))


def _is_array(v: Any) -> bool:
    return isinstance(v, (list, tuple))


def _js_entries(obj: dict) -> list[tuple[str, Any]]:
    """``Object.entries`` order: array-index keys ascending, then the rest in insertion order."""
    index_keys: list[tuple[int, str, Any]] = []
    rest: list[tuple[str, Any]] = []
    for k, v in obj.items():
        key = k if isinstance(k, str) else _js_string(k)
        # <= 10 digits first: int() on a huge digit string is slow, then raises.
        if len(key) <= 10 and _INDEX_KEY.fullmatch(key) and int(key) <= _MAX_ARRAY_INDEX:
            index_keys.append((int(key), key, v))
        else:
            rest.append((key, v))
    if not index_keys:
        return rest
    index_keys.sort(key=lambda t: t[0])
    return [(key, v) for _, key, v in index_keys] + rest


def _estimate_size(v: Any) -> int:
    if v is None:
        return 4
    if isinstance(v, str):
        return _utf16_len(v) + 2
    if isinstance(v, bool):
        return 4 if v else 5
    if isinstance(v, (int, float)):
        return len(_js_string(v))
    if isinstance(v, _BINARY):
        return v.nbytes if isinstance(v, memoryview) else len(v)
    if _is_array(v):
        return 2 + max(0, len(v) - 1)
    if isinstance(v, dict):
        return 2 + sum(_utf16_len(str(k)) + 3 for k in v) + max(0, len(v) - 1)
    return 0


def _normalize(v: Any) -> Any:
    """Map Python-only types onto the JSON shapes the TS port handles, so their text is reviewed.

    Mappings and dataclasses become dicts; sets and other sequences become lists; bytes that
    decode as UTF-8 become text (other bytes stay binary). Anything else is reviewed as ``str(v)``.
    """
    if v is None or isinstance(v, (str, bool, int, float, list, tuple, dict)):
        return v
    if isinstance(v, _BINARY):
        try:
            return bytes(v).decode("utf-8")
        except UnicodeDecodeError:
            return v
    if dataclasses.is_dataclass(v) and not isinstance(v, type):
        return {f.name: getattr(v, f.name) for f in dataclasses.fields(v)}
    if isinstance(v, Mapping):
        return dict(v.items())
    # range is lazy: listing it could allocate without bound.
    if isinstance(v, (AbstractSet, Sequence)) and not isinstance(v, range):
        return list(v)
    return str(v)


def _is_empty_value(v: Any) -> bool:
    if isinstance(v, str):
        return all(len(line) == 0 for line in _LINE_BREAKS.split(v))
    if _is_array(v) or isinstance(v, dict):
        return len(v) == 0
    return False


# --- Budget helpers --------------------------------------------------------------------------


def _item_cap(outer_cap: int, used_now: int, remaining: int) -> int:
    """Budget for one item: everything except a floor per remaining sibling, but at least the floor."""
    avail = outer_cap - used_now
    reserve = min((remaining - 1) * MIN_FIELD_CHARS, max(0, avail - MIN_FIELD_CHARS))
    return outer_cap - reserve


def chunk_overlap(per_chunk: int) -> int:
    return min(CHUNK_OVERLAP_CHARS, per_chunk // 4)


def content_budget(per_chunk: int) -> int:
    return max(1, per_chunk - chunk_overlap(per_chunk) - 1)  # -1 for the "\n" after the carry


def spread_order(n: int, max_out: int | None = None) -> list[int]:
    """Coarse-to-fine visiting order (0, n-1, then halving strides), capped at ``max_out``."""
    if max_out is None:
        max_out = n
    if n <= 2:
        return list(range(n))
    if n > max_out:
        cap = max(1, max_out)
        if cap == 1:
            return [0]
        sample: dict[int, None] = {}
        for j in spread_order(cap):
            sample[(j * (n - 1)) // (cap - 1)] = None
        return list(sample)
    order: list[int] = []
    seen = bytearray(n)

    def push(i: int) -> None:
        if not seen[i]:
            seen[i] = 1
            order.append(i)

    push(0)
    push(n - 1)
    step = n // 2
    while step >= 1:
        for i in range(step, n, step):
            push(i)
        if step == 1:
            break
        step //= 2
    return order


class _Reorder:
    """Reorders child line ranges appended at/after ``base`` by original index (same line set)."""

    def __init__(self, lines: list[str], base: int):
        self._lines = lines
        self._base = base
        self._parts: list[tuple[int, int, int]] = []

    def mark(self, i: int, start: int) -> None:
        if len(self._lines) > start:
            self._parts.append((i, start, len(self._lines)))

    def apply(self) -> None:
        self._parts.sort(key=lambda p: p[0])
        reordered = [self._lines[j] for i, s, e in self._parts for j in range(s, e)]
        del self._lines[self._base :]
        self._lines.extend(reordered)


# --- Serializer ------------------------------------------------------------------------------


class _Meter:
    def __init__(self, limit: int):
        self.bytes = 0
        self.limit = limit
        self.hit = False

    def add(self, v: Any) -> bool:
        """Charge ``v``; False once past the byte budget."""
        if self.hit:
            return False
        self.bytes += _estimate_size(v)
        if self.bytes > self.limit:
            self.hit = True
            return False
        return True


def _sum_string_content(v: Any, depth: int, meter: _Meter) -> int:
    """String-leaf content as the serializer emits it (UTF-16 units, line breaks excluded)."""
    if depth > MAX_TRAVERSAL_DEPTH:
        return 0
    v = _normalize(v)
    if not meter.add(v):
        return 0
    if isinstance(v, str):
        return _utf16_len(v) - len(_LINE_BREAKS.findall(v))
    if _is_array(v):
        items = v
    elif isinstance(v, dict):
        items = v.values()
    else:
        return 0
    total = 0
    for x in items:
        total += _sum_string_content(x, depth + 1, meter)
        if meter.hit:
            break
    return total


class _Serializer:
    def __init__(self, depth_flag: dict[str, Any], max_chars: int, max_size: int):
        self.depth_flag = depth_flag
        self.max_chars = max_chars
        self.meter = _Meter(max_size)
        self.has_string = False
        # A non-string scalar or empty value emitted under an object key: keys are attacker text.
        self.has_keyed_scalar = False
        # A keyed line was dropped for budget: counts as oversize, like a dropped string.
        self.dropped_keyed = False
        self.emitted_string_chars = 0
        self.used = 0
        self.cap = max_chars
        self.non_string_cap = max_chars
        self.reserve_mode = False
        self.budget_truncated = False

    def _degrade(self) -> None:
        self.budget_truncated = True
        self.depth_flag["coverage_degraded"] = True

    def _fits(self, lines: list[str], text: str, limit: int) -> bool:
        sep = 1 if lines else 0
        if self.used + sep + len(text) > limit:
            return False
        lines.append(text)
        self.used += sep + len(text)
        return True

    def _scalar_line(self, lines: list[str], text: str, under_key: bool) -> None:
        if self._fits(lines, text, min(self.cap, self.non_string_cap)):
            if under_key:
                self.has_keyed_scalar = True
        else:
            self._degrade()
            if under_key:
                self.dropped_keyed = True

    def serialize(self, v: Any, prefix: str, lines: list[str], depth: int, keyed: bool, under_key: bool) -> None:
        if self.used >= self.cap:
            self._degrade()
            return
        if depth > MAX_TRAVERSAL_DEPTH:
            self.depth_flag["hit"] = True
            return
        if self.meter.hit:
            return
        v = _normalize(v)
        if not self.meter.add(v):
            self._degrade()
            return
        if v is None or (under_key and _is_empty_value(v)):
            # Keep the key: `{"<text>": null}` / `""` / `{}` / `[]` must not hide `<text>`.
            if under_key:
                self._scalar_line(lines, f"{prefix}: null" if v is None else f"{prefix}:", True)
            return
        if isinstance(v, _BINARY):
            n = v.nbytes if isinstance(v, memoryview) else len(v)
            self._scalar_line(lines, f"{prefix}: <binary {n} bytes>" if prefix else f"<binary {n} bytes>", under_key)
            return
        if _is_array(v):
            if len(v) > ARRAY_SUMMARY_THRESHOLD and all(_is_non_string_scalar(x) for x in v):
                all_number = all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v)
                kind = "numbers" if all_number else "values"
                summary = f"{prefix}: [{len(v)} {kind}]" if prefix else f"[{len(v)} {kind}]"
                self._scalar_line(lines, summary, under_key)
                return
            self._children(len(v), lambda i: (f"{prefix}[{i}]", v[i]), lines, depth, under_key)
            return
        if isinstance(v, dict):
            entries = _js_entries(v)

            def field(i: int) -> tuple[str, Any]:
                # Flatten key line breaks so a `\n` in a key can't forge a line/record boundary.
                key = to_utf16(_LINE_BREAKS_RUN.sub(" ", entries[i][0]))
                return (f"{prefix}.{key}" if prefix else key), entries[i][1]

            self._children(len(entries), field, lines, depth, True)
            return
        self._scalar(v, prefix, lines, keyed, under_key)

    def _children(
        self, n: int, child: Callable[[int], tuple[str, Any]], lines: list[str], depth: int, under_key: bool
    ) -> None:
        """Serialize ``n`` children; ``child(i)`` gives the i-th child's prefix and value."""
        outer_cap = self.cap
        # Reserve pass: visit in spread order (a cutoff drops a spread, not a tail), emit in index order.
        order = spread_order(n, self.max_chars) if self.reserve_mode else None
        total = len(order) if order is not None else n
        parts = _Reorder(lines, len(lines)) if order is not None else None
        for k in range(total):
            if self.meter.hit or self.used >= outer_cap:
                self._degrade()
                break
            idx = order[k] if order is not None else k
            self.cap = _item_cap(outer_cap, self.used, total - k) if self.reserve_mode else outer_cap
            child_prefix, value = child(idx)
            before = len(lines)
            self.serialize(value, child_prefix, lines, depth + 1, True, under_key)
            if parts is not None:
                parts.mark(idx, before)
        self.cap = outer_cap
        if parts is not None:
            parts.apply()

    def _scalar(self, v: Any, prefix: str, lines: list[str], keyed: bool, under_key: bool) -> None:
        is_str = isinstance(v, str)
        s = to_utf16(v if is_str else _js_string(v))
        limit = self.cap if is_str else min(self.cap, self.non_string_cap)
        if not prefix and not keyed:
            # Top-level scalar record: bare lines, blank lines dropped.
            for line in _LINE_BREAKS.split(s):
                if not line:
                    continue
                if self._fits(lines, line, limit):
                    if is_str:
                        self.has_string = True
                        self.emitted_string_chars += len(line)
                elif is_str:
                    sep = 1 if lines else 0
                    room = safe_end(line, limit - self.used - sep)
                    if room > 0:
                        lines.append(line[:room])
                        self.used += sep + room
                        self.has_string = True
                        self.emitted_string_chars += room
                    self._degrade()
                    break
                else:
                    break
            return
        # Per-line `field:` prefix so a multi-line value can't emit a bare line or forge a boundary.
        for line in _LINE_BREAKS.split(s):
            if not line:
                continue
            full = f"{prefix}: {line}"
            if self._fits(lines, full, limit):
                if is_str:
                    self.has_string = True
                    self.emitted_string_chars += len(line)
                elif under_key:
                    self.has_keyed_scalar = True
            elif is_str:
                # Truncate to the remaining share, keeping the `key:` prefix.
                sep = 1 if lines else 0
                room = safe_end(full, self.cap - self.used - sep)
                if room > len(prefix) + 2:
                    lines.append(full[:room])
                    self.used += sep + room
                    self.has_string = True
                    self.emitted_string_chars += room - len(prefix) - 2
                self._degrade()
                break
            else:
                self._degrade()
                if under_key:
                    self.dropped_keyed = True
                break

    def run_records(self, records: list[Any], top_is_array: bool) -> list[str]:
        n = len(records)
        order = spread_order(n, self.max_chars) if self.reserve_mode else None
        total = len(order) if order is not None else n
        collected: list[tuple[int, str]] = []
        emitted = 0
        for oi in range(total):
            if self.meter.hit or self.used + (2 if emitted else 0) >= self.max_chars:
                self._degrade()
                break
            i = order[oi] if order is not None else oi
            used_before = self.used
            self.cap = _item_cap(self.max_chars, self.used, total - oi) if self.reserve_mode else self.max_chars
            self.non_string_cap = (
                min(used_before + int((self.cap - used_before) * SCALAR_BUDGET_FRACTION), self.cap)
                if self.reserve_mode
                else self.cap
            )
            self.used += 2 if emitted else 0  # the "\n\n" before this block (rolled back if empty)
            record = _normalize(records[i])
            lines: list[str] = []
            prefix = f"[{i}]" if top_is_array and not isinstance(record, dict) else ""
            self.serialize(record, prefix, lines, 0, False, False)
            if lines:
                collected.append((i, "\n".join(lines)))
                emitted += 1
            else:
                self.used = used_before
        collected.sort(key=lambda c: c[0])
        return [block for _, block in collected]


def format_records_for_tier3(value: Any, depth_flag: dict[str, Any], max_chars: int, max_size: int) -> str:
    """Serialize a tool result as record-oriented ``field: value`` blocks, within ``max_chars``.

    Returns the text in a UTF-16 view (see :func:`to_utf16`); ``""`` when nothing reviewable was
    emitted. Sets ``depth_flag["coverage_degraded"]`` on any drop and ``depth_flag["budget_exceeded"]``
    when content didn't fit (oversize).
    """
    value = _normalize(value)
    top_is_array = _is_array(value)
    records = list(value) if top_is_array else [value]
    ser = _Serializer(depth_flag, max_chars, max_size)
    size_limit_hit = False

    blocks = ser.run_records(records, top_is_array)  # greedy
    size_limit_hit |= ser.meter.hit
    if ser.budget_truncated:
        # Something didn't fit at full budget: redo with a per-sibling reserve and spread sampling.
        fresh = _Serializer(depth_flag, max_chars, max_size)
        fresh.reserve_mode = True
        depth_flag.pop("coverage_degraded", None)
        ser = fresh
        blocks = ser.run_records(records, top_is_array)
        size_limit_hit |= ser.meter.hit
    # Definitive oversize check: string content emitted vs the input's string content.
    meter = _Meter(max_size)
    input_chars = 0
    for r in records:
        input_chars += _sum_string_content(r, 0, meter)
        if meter.hit:
            break
    size_limit_hit |= meter.hit
    if ser.emitted_string_chars < input_chars or size_limit_hit or ser.dropped_keyed:
        depth_flag["budget_exceeded"] = True
    return "\n\n".join(blocks) if ser.has_string or ser.has_keyed_scalar else ""


# --- Chunking --------------------------------------------------------------------------------


def _split_long_line(line: str, max_len: int, overlap: int) -> list[str]:
    """Split into <= max_len word-aligned pieces, each overlapping the previous by ``overlap``."""
    out: list[str] = []
    start = 0
    while start < len(line):
        end = min(start + max_len, len(line))
        if end < len(line):
            sp = line.rfind(" ", 0, end + 1)
            if sp > start:
                end = sp
            else:
                end = safe_end(line, end)
                if end <= start:
                    end = min(len(line), start + 2)  # budget of 1 at a pair: take the pair
        out.append(line[start:end])
        if end >= len(line):
            break
        start = safe_start(line, max(end - overlap, start + 1))
    return out


def chunk_tier3_input(text: str, per_chunk: int) -> list[str]:
    """Chunk the serialized (UTF-16 view) input; each chunk carries the previous chunk's tail.

    Returns ordinary Python strings.
    """
    if len(text) <= per_chunk:
        return [from_utf16(text)]
    overlap = chunk_overlap(per_chunk)
    budget = content_budget(per_chunk)
    units: list[str] = []
    for line in text.split("\n"):
        if len(line) <= budget:
            units.append(line)
        else:
            units.extend(_split_long_line(line, budget, overlap))
    chunks: list[str] = []
    i = 0
    carry = ""
    while i < len(units):
        content = ""
        while i < len(units):
            add = units[i] if not content else f"\n{units[i]}"
            if content and len(content) + len(add) > budget:
                break
            content += add
            i += 1
        if not content:
            content = units[i]
            i += 1
        chunk = f"{carry}\n{content}" if carry else content
        chunks.append(chunk)
        carry = chunk[safe_start(chunk, max(0, len(chunk) - overlap)) :]
    return [from_utf16(c) for c in chunks]
