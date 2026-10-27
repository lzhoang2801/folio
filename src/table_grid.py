from __future__ import annotations

from dataclasses import dataclass

from bs4 import BeautifulSoup, Tag

MAX_HEADER_ROWS = 5


def _colspan(cell: Tag) -> int:
    return _span(cell, "colspan")


def _rowspan(cell: Tag) -> int:
    return _span(cell, "rowspan")


def _span(cell: Tag, name: str) -> int:
    value = cell.get(name, "1")
    if not isinstance(value, str) or not value.isdecimal() or not 1 <= int(value) <= 128:
        raise ValueError("Invalid or oversized table span")
    return int(value)


def full_to_half(text: str) -> str:
    return "".join(
        chr(ord(char) - 0xFEE0) if 0xFF01 <= ord(char) <= 0xFF5E else char for char in text
    )


@dataclass
class RowMetrics:
    row_idx: int
    effective_cols: int
    actual_cols: int
    visual_cols: int


@dataclass
class RowSignature:
    effective_cols: int
    colspans: tuple[int, ...]
    rowspans: tuple[int, ...]
    normalized_texts: tuple[str, ...]
    display_texts: tuple[str, ...]

    @property
    def cell_count(self) -> int:
        """Number of explicit cells in the row signature."""
        return len(self.colspans)


@dataclass
class RenderedCellSegment:
    text: str
    start_col: int
    end_col: int


@dataclass
class RowScanResult:
    row_effective_cols: list[int]
    row_metrics: list[RowMetrics]
    total_cols: int
    last_nonempty_row_metrics: RowMetrics | None
    tail_occupied: dict[int, set[int]]


@dataclass
class TableMergeState:
    soup: BeautifulSoup
    tbody: Tag
    rows: list[Tag]
    total_cols: int
    front_header_info: list[RowSignature]
    front_first_data_row_metrics: dict[int, RowMetrics]
    last_data_row_metrics: RowMetrics | None
    row_effective_cols: list[int]
    tail_occupied: dict[int, set[int]]


def _normalize_cell_text(cell: Tag) -> str:
    return "".join(full_to_half(cell.get_text()).split())


def _display_cell_text(cell: Tag) -> str:
    return full_to_half(cell.get_text().strip())


def _scan_rows(
    rows: list[Tag], initial_occupied: dict[int, set[int]] | None = None, start_row_idx: int = 0
) -> RowScanResult:
    occupied: dict[int, dict[int, bool]] = {}
    max_cols = 0
    for row_offset, cols in (initial_occupied or {}).items():
        if not cols:
            continue
        occupied[row_offset] = dict.fromkeys(cols, True)
        max_cols = max(max_cols, max(cols) + 1)
    row_effective_cols: list[int] = []
    row_metrics: list[RowMetrics] = []
    last_nonempty_row_metrics: RowMetrics | None = None
    for local_idx, row in enumerate(rows):
        occupied_row = occupied.setdefault(local_idx, {})
        col_idx = 0
        cells = row.find_all(["td", "th"])
        actual_cols = 0
        for cell in cells:
            while col_idx in occupied_row:
                col_idx += 1
            colspan = _colspan(cell)
            rowspan = _rowspan(cell)
            actual_cols += colspan
            if col_idx + colspan > 128:
                raise ValueError("Table grid exceeds 128 logical columns")
            for row_offset in range(rowspan):
                target_idx = local_idx + row_offset
                occupied_target = occupied.setdefault(target_idx, {})
                for col in range(col_idx, col_idx + colspan):
                    occupied_target[col] = True
            col_idx += colspan
            max_cols = max(max_cols, col_idx)
        effective_cols = max(occupied_row.keys()) + 1 if occupied_row else 0
        row_effective_cols.append(effective_cols)
        max_cols = max(max_cols, effective_cols)
        metrics = RowMetrics(
            row_idx=start_row_idx + local_idx,
            effective_cols=effective_cols,
            actual_cols=actual_cols,
            visual_cols=len(cells),
        )
        row_metrics.append(metrics)
        if cells:
            last_nonempty_row_metrics = metrics
    tail_occupied = {
        row_idx - len(rows): set(cols.keys())
        for row_idx, cols in occupied.items()
        if row_idx >= len(rows) and cols
    }
    return RowScanResult(
        row_effective_cols=row_effective_cols,
        row_metrics=row_metrics,
        total_cols=max_cols,
        last_nonempty_row_metrics=last_nonempty_row_metrics,
        tail_occupied=tail_occupied,
    )


def _build_row_signature(row: Tag, effective_cols: int) -> RowSignature:
    cells = row.find_all(["td", "th"])
    return RowSignature(
        effective_cols=effective_cols,
        colspans=tuple((_colspan(cell) for cell in cells)),
        rowspans=tuple((_rowspan(cell) for cell in cells)),
        normalized_texts=tuple((_normalize_cell_text(cell) for cell in cells)),
        display_texts=tuple((_display_cell_text(cell) for cell in cells)),
    )


def _build_front_cache(
    rows: list[Tag], max_header_rows: int = MAX_HEADER_ROWS
) -> tuple[list[RowSignature], dict[int, RowMetrics]]:
    front_limit = min(len(rows), max_header_rows + 1)
    front_rows = rows[:front_limit]
    front_scan = _scan_rows(front_rows)
    front_header_info = [
        _build_row_signature(
            front_rows[idx], front_scan.row_effective_cols[idx])
        for idx in range(min(len(front_rows), max_header_rows))
    ]
    front_first_data_row_metrics = dict(enumerate(front_scan.row_metrics))
    return (front_header_info, front_first_data_row_metrics)


def _refresh_table_state_metrics(state: TableMergeState) -> None:
    scan = _scan_rows(state.rows)
    state.row_effective_cols = scan.row_effective_cols
    state.total_cols = scan.total_cols
    state.last_data_row_metrics = scan.last_nonempty_row_metrics
    state.tail_occupied = scan.tail_occupied
    state.front_header_info, state.front_first_data_row_metrics = _build_front_cache(
        state.rows)


def _scan_row_visual_sources(
    rows: list[Tag], target_row_index: int, initial_occupied: dict[int, set[int]] | None = None
) -> tuple[dict[int, tuple[int, int]], int]:
    if target_row_index < 0:
        target_row_index += len(rows)
    if target_row_index < 0 or target_row_index >= len(rows):
        return ({}, 0)
    occupied: dict[int, dict[int, tuple[int, int]]] = {}
    total_cols = 0
    for row_offset, cols in (initial_occupied or {}).items():
        if not cols:
            continue
        occupied[row_offset] = {col: (-1, col) for col in cols}
        total_cols = max(total_cols, max(cols) + 1)
    for r_idx in range(target_row_index + 1):
        occupied_row = occupied.setdefault(r_idx, {})
        col_idx = 0
        cells = rows[r_idx].find_all(["td", "th"])
        for cell_idx, cell in enumerate(cells):
            while col_idx in occupied_row:
                col_idx += 1
            colspan = _colspan(cell)
            rowspan = _rowspan(cell)
            source_marker = (r_idx, cell_idx)
            for ro in range(rowspan):
                target_idx = r_idx + ro
                occ = occupied.setdefault(target_idx, {})
                for c in range(col_idx, col_idx + colspan):
                    occ[c] = source_marker
            col_idx += colspan
            total_cols = max(total_cols, col_idx)
    return (occupied.get(target_row_index, {}), total_cols)


def build_visual_col_mapping(
    rows: list[Tag], target_row_index: int, initial_occupied: dict[int, set[int]] | None = None
) -> list[int]:
    if target_row_index < 0:
        target_row_index += len(rows)
    if target_row_index < 0 or target_row_index >= len(rows):
        return []
    target_occupied, _ = _scan_row_visual_sources(
        rows, target_row_index, initial_occupied=initial_occupied
    )
    col_idx = 0
    mapping = []
    target_cells = rows[target_row_index].find_all(["td", "th"])
    for cell in target_cells:
        while col_idx in target_occupied and target_occupied[col_idx][0] < target_row_index:
            col_idx += 1
        mapping.append(col_idx)
        colspan = _colspan(cell)
        col_idx += colspan
    return mapping


def build_row_rendered_cell_segments(
    rows: list[Tag], target_row_index: int, initial_occupied: dict[int, set[int]] | None = None
) -> list[RenderedCellSegment]:
    if target_row_index < 0:
        target_row_index += len(rows)
    if target_row_index < 0 or target_row_index >= len(rows):
        return []
    target_occupied, total_cols = _scan_row_visual_sources(
        rows, target_row_index, initial_occupied=initial_occupied
    )
    if total_cols == 0:
        return []
    segments: list[RenderedCellSegment] = []
    current_marker: tuple[int, int] | None = None
    current_start_col: int | None = None
    current_text = ""
    for col_idx in range(total_cols):
        marker = target_occupied.get(col_idx)
        if marker is None:
            if current_marker is not None and current_start_col is not None:
                segments.append(
                    RenderedCellSegment(
                        text=current_text, start_col=current_start_col, end_col=col_idx
                    )
                )
            current_marker = None
            current_start_col = None
            current_text = ""
            continue
        if marker != current_marker:
            if current_marker is not None and current_start_col is not None:
                segments.append(
                    RenderedCellSegment(
                        text=current_text, start_col=current_start_col, end_col=col_idx
                    )
                )
            current_marker = marker
            current_start_col = col_idx
            source_row_idx, source_cell_idx = marker
            current_text = ""
            if source_row_idx >= 0:
                source_cells = rows[source_row_idx].find_all(["td", "th"])
                if source_cell_idx < len(source_cells):
                    current_text = _display_cell_text(
                        source_cells[source_cell_idx])
    if current_marker is not None and current_start_col is not None:
        segments.append(
            RenderedCellSegment(
                text=current_text, start_col=current_start_col, end_col=total_cols)
        )
    return segments


def calculate_row_rendered_segments(rows: list[Tag], target_row_index: int) -> int:
    target_occupied, total_cols = _scan_row_visual_sources(
        rows, target_row_index)
    if total_cols == 0:
        return 0
    segment_count = 0
    previous_marker: tuple[int, int] | None = None
    for col_idx in range(total_cols):
        marker = target_occupied.get(col_idx)
        if marker is None:
            previous_marker = None
            continue
        if marker != previous_marker:
            segment_count += 1
            previous_marker = marker
    return segment_count


@dataclass(frozen=True)
class _TableCellPlacement:
    cell: Tag
    row_index: int
    start_col: int
    end_col: int


@dataclass(frozen=True)
class _TableGridSnapshot:
    cells: list[_TableCellPlacement]
    row_owners: list[dict[int, _TableCellPlacement]]


@dataclass(frozen=True)
class _TableColumnMapping:
    source_columns: int
    target_columns: int
    boundaries: dict[int, int]


def _snapshot_table_grid(rows: list[Tag], total_columns: int) -> _TableGridSnapshot:
    if total_columns <= 0 or not rows:
        raise ValueError("Table grid must contain rows and columns")
    occupied: dict[int, dict[int, _TableCellPlacement]] = {}
    placements: list[_TableCellPlacement] = []
    row_owners: list[dict[int, _TableCellPlacement]] = []
    expected_columns = set(range(total_columns))
    for row_index, row in enumerate(rows):
        occupied_row = occupied.setdefault(row_index, {})
        column = 0
        for cell in row.find_all(["td", "th"], recursive=False):
            if cell.find("table") is not None:
                raise ValueError(
                    "Nested table grids cannot be aligned across pages")
            while column in occupied_row:
                column += 1
            colspan = _colspan(cell)
            rowspan = _rowspan(cell)
            if colspan <= 0 or rowspan <= 0 or column + colspan > total_columns:
                raise ValueError("Invalid table cell span")
            placement = _TableCellPlacement(
                cell, row_index, column, column + colspan)
            for offset in range(rowspan):
                occupied_target = occupied.setdefault(row_index + offset, {})
                for covered_column in range(placement.start_col, placement.end_col):
                    if covered_column in occupied_target:
                        raise ValueError("Table cells overlap")
                    occupied_target[covered_column] = placement
            placements.append(placement)
            column = placement.end_col
        if set(occupied_row) != expected_columns:
            raise ValueError("Table rows have inconsistent logical columns")
        row_owners.append(occupied_row)
    return _TableGridSnapshot(placements, row_owners)


def _identity_column_mapping(columns: int) -> _TableColumnMapping:
    return _TableColumnMapping(columns, columns, dict(zip(range(columns + 1), range(columns + 1))))


def _boundary_column_mapping(
    source: TableMergeState, source_row_index: int, target: TableMergeState, target_row_index: int
) -> _TableColumnMapping:
    source_segments = build_row_rendered_cell_segments(
        source.rows, source_row_index)
    target_segments = build_row_rendered_cell_segments(
        target.rows, target_row_index)
    if not source_segments or len(source_segments) != len(target_segments):
        raise ValueError("Table boundary segments cannot be paired")
    boundaries = {0: 0}
    source_end = target_end = 0
    for source_segment, target_segment in zip(source_segments, target_segments):
        if source_segment.start_col != source_end or target_segment.start_col != target_end:
            raise ValueError("Table boundary segments contain gaps")
        source_end = source_segment.end_col
        target_end = target_segment.end_col
        if source_end <= source_segment.start_col or target_end <= target_segment.start_col:
            raise ValueError("Table boundary segments contain invalid spans")
        boundaries[source_end] = target_end
    if source_end != source.total_cols or target_end != target.total_cols:
        raise ValueError("Table boundary does not cover its grid")
    if source.total_cols == target.total_cols:
        if any(
            (source_column != target_column for source_column,
             target_column in boundaries.items())
        ):
            raise ValueError(
                "Equal-width table grids have different cell boundaries")
        return _identity_column_mapping(source.total_cols)
    return _TableColumnMapping(source.total_cols, target.total_cols, boundaries)


def _align_table_columns(
    previous: TableMergeState, current: TableMergeState, header_count: int
) -> tuple[_TableColumnMapping, _TableColumnMapping]:
    previous_grid = _snapshot_table_grid(previous.rows, previous.total_cols)
    current_grid = _snapshot_table_grid(current.rows, current.total_cols)
    if previous.total_cols >= current.total_cols:
        previous_mapping = _identity_column_mapping(previous.total_cols)
        current_mapping = _boundary_column_mapping(
            current, header_count, previous, len(previous.rows) - 1
        )
    else:
        previous_mapping = _boundary_column_mapping(
            previous, len(previous.rows) - 1, current, header_count
        )
        current_mapping = _identity_column_mapping(current.total_cols)
    for grid, mapping in ((previous_grid, previous_mapping), (current_grid, current_mapping)):
        for placement in grid.cells:
            if (
                placement.start_col not in mapping.boundaries
                or placement.end_col not in mapping.boundaries
            ):
                raise ValueError(
                    "Table contains an unmapped internal column boundary")
        for placement in grid.cells:
            mapped_span = (
                mapping.boundaries[placement.end_col] -
                mapping.boundaries[placement.start_col]
            )
            if mapped_span != placement.end_col - placement.start_col:
                placement.cell["colspan"] = str(mapped_span)
    return (previous_mapping, current_mapping)


def _compose_column_mappings(
    source_mapping: _TableColumnMapping, target_mapping: _TableColumnMapping
) -> _TableColumnMapping:
    if source_mapping.target_columns != target_mapping.source_columns:
        raise ValueError("Table grid mapping dimensions do not match")
    return _TableColumnMapping(
        source_mapping.source_columns,
        target_mapping.target_columns,
        {
            source: target_mapping.boundaries[target]
            for source, target in source_mapping.boundaries.items()
            if target in target_mapping.boundaries
        },
    )


def _project_cell_merge_flags(flags: object, mapping: _TableColumnMapping) -> list[int] | None:
    if not isinstance(flags, list) or len(flags) != mapping.source_columns:
        return None
    if any((type(flag) is not int or flag not in (0, 1) for flag in flags)):
        return None
    boundaries = sorted(mapping.boundaries)
    if not boundaries or boundaries[0] != 0 or boundaries[-1] != mapping.source_columns:
        return None
    projected = [0] * mapping.target_columns
    for start, end in zip(boundaries, boundaries[1:]):
        interval_flags = flags[start:end]
        if not interval_flags or any((flag != interval_flags[0] for flag in interval_flags)):
            return None
        target_start = mapping.boundaries[start]
        target_end = mapping.boundaries[end]
        if not 0 <= target_start < target_end <= mapping.target_columns:
            return None
        projected[target_start:target_end] = [
            interval_flags[0]] * (target_end - target_start)
    return projected
