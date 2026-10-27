from __future__ import annotations

import math
from copy import deepcopy

from bs4 import BeautifulSoup, Tag

from .table_grid import (
    MAX_HEADER_ROWS,
    TableMergeState,
    _align_table_columns,
    _build_front_cache,
    _colspan,
    _compose_column_mappings,
    _identity_column_mapping,
    _project_cell_merge_flags,
    _refresh_table_state_metrics,
    _rowspan,
    _scan_rows,
    _snapshot_table_grid,
    _TableColumnMapping,
    build_row_rendered_cell_segments,
    build_visual_col_mapping,
    calculate_row_rendered_segments,
)

__all__ = [
    "build_table_state_from_html",
    "detect_table_headers",
    "can_merge_by_structure",
    "table_widths_are_compatible",
    "expand_header_count_by_rowspan",
    "build_row_rendered_cell_segments",
    "merge_table_fragments",
]


def build_table_state_from_html(html: str) -> TableMergeState | None:
    if not html:
        return None
    soup = BeautifulSoup(html, "html.parser")
    tables = soup.find_all("table")
    if len(tables) != 1:
        return None
    table = tables[0]
    if any(str(part).strip() for part in soup.contents if part is not table):
        return None
    rows = table.find_all("tr")
    if not rows or table.find("tfoot") is not None:
        return None
    if any(row.parent.name not in {"table", "thead", "tbody"} for row in rows):
        return None
    if any(cell.parent.name != "tr" for cell in table.find_all(["td", "th"])):
        return None
    if any(
        str(node).strip() and node.find_parent(["td", "th", "caption"]) is None
        for node in table.find_all(string=True)
    ):
        return None
    try:
        scan = _scan_rows(rows)
        if scan.total_cols <= 0 or scan.last_nonempty_row_metrics is None:
            return None
        front, first_data = _build_front_cache(rows)
    except (AssertionError, TypeError, ValueError):
        return None
    return TableMergeState(
        soup=soup,
        tbody=table.find("tbody", recursive=False) or table,
        rows=rows,
        total_cols=scan.total_cols,
        front_header_info=front,
        front_first_data_row_metrics=first_data,
        last_data_row_metrics=scan.last_nonempty_row_metrics,
        row_effective_cols=scan.row_effective_cols,
        tail_occupied=scan.tail_occupied,
    )


def table_widths_are_compatible(previous_bbox, current_bbox) -> bool:
    for bbox in (previous_bbox, current_bbox):
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return False
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in bbox
        ):
            return False
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            return False
    try:
        left = previous_bbox[2] - previous_bbox[0]
        right = current_bbox[2] - current_bbox[0]
        narrow = min(left, right)
        return narrow > 0 and abs(left - right) / narrow < 0.1
    except (IndexError, TypeError, ZeroDivisionError):
        return False


def can_merge_by_structure(current: TableMergeState, previous: TableMergeState) -> bool:
    if (
        current.total_cols <= 0
        or previous.total_cols <= 0
        or current.last_data_row_metrics is None
        or previous.last_data_row_metrics is None
    ):
        return False
    return previous.total_cols == current.total_cols or check_rows_match(previous, current)


def _declared_header_count(state: TableMergeState) -> int:
    count = 0
    for row in state.rows:
        cells = row.find_all(["th", "td"], recursive=False)
        if row.find_parent("thead") is not None or (
            cells and all(cell.name == "th" for cell in cells)
        ):
            count += 1
        else:
            break
    return _expand_header_count_by_rowspan(state.rows, count)


def _headers_prove_identity(
    previous: TableMergeState,
    current: TableMergeState,
    header_count: int,
    cell_merge: object,
    source_columns: _TableColumnMapping,
) -> bool:
    if previous.total_cols != current.total_cols or header_count <= 0:
        return False
    if (
        _declared_header_count(previous) != header_count
        or _declared_header_count(current) != header_count
    ):
        return False
    if cell_merge is not None:
        projected = _project_cell_merge_flags(cell_merge, source_columns)
        if projected is None or any(projected):
            return False
    expected_boundaries = set(range(previous.total_cols + 1))
    for state in (previous, current):
        grid = _snapshot_table_grid(state.rows, state.total_cols)
        boundaries = {
            boundary
            for cell in grid.cells
            if cell.row_index < header_count
            for boundary in (cell.start_col, cell.end_col)
        }
        if boundaries != expected_boundaries:
            return False
    return True


def merge_table_fragments(
    previous_html: str,
    current_html: str,
    *,
    cell_merge: list[int] | None = None,
    previous_source_columns: _TableColumnMapping | None = None,
    skip_header_rows: int | None = None,
) -> tuple[str | None, _TableColumnMapping | str]:
    previous = build_table_state_from_html(previous_html)
    current = build_table_state_from_html(current_html)
    if previous is None or current is None:
        return None, "invalid or unsupported HTML table"
    if current.soup.find("caption") is not None:
        return None, "current caption must be resolved before content merging"
    try:
        if not can_merge_by_structure(current, previous):
            raise ValueError("Table boundary structures are incompatible")
        header_count = (
            detect_table_headers(previous, current)[0]
            if skip_header_rows is None
            else skip_header_rows
        )
        if type(header_count) is not int or not 0 <= header_count <= len(current.rows):
            raise ValueError("Invalid repeated-header row count")
        header_count = _expand_header_count_by_rowspan(
            current.rows, header_count)
        if header_count >= len(current.rows):
            raise ValueError("Current table contains only repeated headers")
        rows1, rows2 = previous.rows, current.rows
        if previous_source_columns is not None and not isinstance(
            previous_source_columns, _TableColumnMapping
        ):
            raise ValueError("Invalid previous source-column mapping")
        source_columns = previous_source_columns or _identity_column_mapping(
            previous.total_cols)
        try:
            previous_mapping, current_mapping = _align_table_columns(
                previous, current, header_count
            )
        except ValueError:
            if not _headers_prove_identity(
                previous, current, header_count, cell_merge, source_columns
            ):
                raise
            previous_mapping = current_mapping = _identity_column_mapping(
                previous.total_cols)
        source_to_merged = _compose_column_mappings(
            source_columns, previous_mapping)
        projected_flags = _project_cell_merge_flags(
            cell_merge, source_to_merged)
        _refresh_table_state_metrics(previous)
        _refresh_table_state_metrics(current)
        target_columns = previous_mapping.target_columns
        _clip_overlapped_blank_rowspan_cells(
            rows2[header_count:], previous.tail_occupied)
        _snapshot_table_grid([*rows1, *rows2[header_count:]], target_columns)
        applied = _apply_cell_merge(
            previous, current, header_count, projected_flags)
        appended = rows2[header_count:]
        for row in appended:
            previous.tbody.append(row.extract())
        if not appended and not applied:
            raise ValueError("No current table content remains to merge")
        previous.rows.extend(appended)
        _snapshot_table_grid(previous.rows, target_columns)
        _refresh_table_state_metrics(previous)
        if previous.total_cols != target_columns:
            raise ValueError("Merged table changed its aligned column count")
    except (AssertionError, KeyError, TypeError, ValueError) as error:
        return None, str(error)
    return str(previous.soup), current_mapping


def detect_table_headers(
    state1: TableMergeState, state2: TableMergeState, max_header_rows: int = MAX_HEADER_ROWS
) -> tuple[int, bool, list[list[str]]]:
    front_rows1 = state1.front_header_info[:max_header_rows]
    front_rows2 = state2.front_header_info[:max_header_rows]
    min_rows = min(len(front_rows1), len(front_rows2), max_header_rows)
    header_rows = 0
    headers_match = True
    header_texts = []
    for row_idx in range(min_rows):
        row1 = front_rows1[row_idx]
        row2 = front_rows2[row_idx]
        structure_match = (
            row1.cell_count == row2.cell_count
            and row1.effective_cols == row2.effective_cols
            and (row1.colspans == row2.colspans)
            and (row1.rowspans == row2.rowspans)
            and (row1.normalized_texts == row2.normalized_texts)
        )
        if structure_match:
            header_rows += 1
            header_texts.append(list(row1.display_texts))
        else:
            headers_match = header_rows > 0
            break
    if header_rows == 0:
        header_rows, headers_match, header_texts = _detect_table_headers_visual(
            state1, state2, max_header_rows=max_header_rows
        )
    for row_index in range(header_rows):
        if _header_media(state1.rows[row_index]) != _header_media(state2.rows[row_index]):
            header_rows = row_index
            headers_match = header_rows > 0
            header_texts = header_texts[:header_rows]
            break
    return (header_rows, headers_match, header_texts)


def _header_media(row: Tag) -> list[str]:
    return [
        str(tag)
        for tag in row.find_all(
            ["a", "img", "svg", "math", "eq", "figure", "object", "embed", "canvas"]
        )
    ]


def _detect_table_headers_visual(
    state1: TableMergeState, state2: TableMergeState, max_header_rows: int = MAX_HEADER_ROWS
) -> tuple[int, bool, list[list[str]]]:
    front_rows1 = state1.front_header_info[:max_header_rows]
    front_rows2 = state2.front_header_info[:max_header_rows]
    min_rows = min(len(front_rows1), len(front_rows2), max_header_rows)
    header_rows = 0
    headers_match = True
    header_texts = []
    for row_idx in range(min_rows):
        row1 = front_rows1[row_idx]
        row2 = front_rows2[row_idx]
        rendered_segments1 = calculate_row_rendered_segments(
            state1.rows, row_idx)
        rendered_segments2 = calculate_row_rendered_segments(
            state2.rows, row_idx)
        if (
            row1.normalized_texts == row2.normalized_texts
            and rendered_segments1 == rendered_segments2
        ):
            header_rows += 1
            header_texts.append(list(row1.display_texts))
        else:
            headers_match = header_rows > 0
            break
    if header_rows == 0:
        headers_match = False
    return (header_rows, headers_match, header_texts)


def _expand_header_count_by_rowspan(rows: list[Tag], header_count: int) -> int:
    if header_count <= 0 or not rows:
        return header_count
    expanded_header_count = min(header_count, len(rows))
    row_idx = 0
    while row_idx < expanded_header_count:
        row = rows[row_idx]
        for cell in row.find_all(["td", "th"]):
            rowspan = _rowspan(cell)
            if rowspan > 1:
                expanded_header_count = max(
                    expanded_header_count, row_idx + rowspan)
                expanded_header_count = min(expanded_header_count, len(rows))
        row_idx += 1
    return expanded_header_count


expand_header_count_by_rowspan = _expand_header_count_by_rowspan


def check_rows_match(previous_state: TableMergeState, current_state: TableMergeState) -> bool:
    last_row_metrics = previous_state.last_data_row_metrics
    if last_row_metrics is None:
        return False
    header_count, _, _ = detect_table_headers(previous_state, current_state)
    header_count = _expand_header_count_by_rowspan(
        current_state.rows, header_count)
    first_data_row_metrics = current_state.front_first_data_row_metrics.get(
        header_count)
    if first_data_row_metrics is None:
        return False
    previous_rendered_segments = calculate_row_rendered_segments(
        previous_state.rows, last_row_metrics.row_idx
    )
    current_rendered_segments = calculate_row_rendered_segments(
        current_state.rows, first_data_row_metrics.row_idx
    )
    return (
        last_row_metrics.effective_cols == first_data_row_metrics.effective_cols
        or last_row_metrics.actual_cols == first_data_row_metrics.actual_cols
        or previous_rendered_segments == current_rendered_segments
    )


def _cell_has_semantic_content(cell: Tag) -> bool:
    if cell.get_text(strip=True):
        return True
    return (
        cell.find(["img", "svg", "math", "eq", "table",
                  "figure", "object", "embed", "canvas"])
        is not None
    )


def _row_has_semantic_content(row: Tag) -> bool:
    return any((_cell_has_semantic_content(cell) for cell in row.find_all(["td", "th"])))


def _insert_cell_before_visual_column(
    rows: list[Tag], target_row_index: int, start_vcol: int, cell: Tag
) -> None:
    target_row = rows[target_row_index]
    target_cells = target_row.find_all(["td", "th"])
    target_vcol_map = build_visual_col_mapping(rows, target_row_index)
    for idx, target_start_vcol in enumerate(target_vcol_map):
        if target_start_vcol >= start_vcol:
            target_cells[idx].insert_before(cell)
            return
    target_row.append(cell)


def _carry_rowspan_structure_to_next_row(rows: list[Tag], row_idx: int) -> None:
    next_row_idx = row_idx + 1
    if next_row_idx >= len(rows):
        return
    current_row = rows[row_idx]
    current_cells = current_row.find_all(["td", "th"])
    current_vcol_map = build_visual_col_mapping(rows, row_idx)
    carried_cells = []
    for cell, start_vcol in zip(current_cells, current_vcol_map):
        rowspan = _rowspan(cell)
        if rowspan <= 1 or _cell_has_semantic_content(cell):
            continue
        carried_cell = deepcopy(cell)
        new_rowspan = rowspan - 1
        if new_rowspan > 1:
            carried_cell["rowspan"] = str(new_rowspan)
        else:
            carried_cell.attrs.pop("rowspan", None)
        carried_cells.append((start_vcol, carried_cell))
    for start_vcol, carried_cell in sorted(carried_cells, key=lambda item: item[0], reverse=True):
        _insert_cell_before_visual_column(
            rows, next_row_idx, start_vcol, carried_cell)


def _clip_overlapped_blank_rowspan_cells(
    rows: list[Tag], initial_occupied: dict[int, set[int]]
) -> bool:
    if not rows or not initial_occupied:
        return False
    cells_to_remove = []
    cells_to_move = []
    for row_idx, row in enumerate(rows):
        cells = row.find_all(["td", "th"])
        visual_col_map = build_visual_col_mapping(rows, row_idx)
        for cell, start_vcol in zip(cells, visual_col_map):
            rowspan = _rowspan(cell)
            if rowspan <= 1 or _cell_has_semantic_content(cell):
                continue
            colspan = _colspan(cell)
            occupied_cols = set(range(start_vcol, start_vcol + colspan))
            if not occupied_cols:
                continue
            overlap_rows = 0
            while overlap_rows < rowspan:
                covered_cols = initial_occupied.get(
                    row_idx + overlap_rows, set())
                if not occupied_cols.issubset(covered_cols):
                    break
                overlap_rows += 1
            if overlap_rows == 0:
                continue
            remaining_rowspan = rowspan - overlap_rows
            target_row_idx = row_idx + overlap_rows
            if remaining_rowspan > 0 and target_row_idx >= len(rows):
                continue
            cells_to_remove.append(cell)
            if remaining_rowspan > 0:
                moved_cell = deepcopy(cell)
                if remaining_rowspan > 1:
                    moved_cell["rowspan"] = str(remaining_rowspan)
                else:
                    moved_cell.attrs.pop("rowspan", None)
                cells_to_move.append((target_row_idx, start_vcol, moved_cell))
    if not cells_to_remove:
        return False
    for cell in cells_to_remove:
        cell.extract()
    for target_row_idx, start_vcol, moved_cell in sorted(
        cells_to_move, key=lambda item: (item[0], item[1]), reverse=True
    ):
        _insert_cell_before_visual_column(
            rows, target_row_idx, start_vcol, moved_cell)
    return True


def _apply_cell_merge(
    previous_state: TableMergeState,
    current_state: TableMergeState,
    header_count: int,
    cell_merge: list[int] | None,
) -> bool:
    if cell_merge is None or len(cell_merge) != previous_state.total_cols:
        return False
    rows2 = current_state.rows
    if header_count >= len(rows2):
        return False
    if not previous_state.rows:
        return False
    first_data_row = rows2[header_count]
    previous_row_count = len(previous_state.rows)
    combined_grid = _snapshot_table_grid(
        [*previous_state.rows, *rows2[header_count:]], previous_state.total_cols
    )
    previous_owners = combined_grid.row_owners[previous_row_count - 1]
    transferred_sources: set[int] = set()
    transferred_targets: set[int] = set()
    for source in combined_grid.cells:
        if source.row_index != previous_row_count or not _cell_has_semantic_content(source.cell):
            continue
        if any((flag != 1 for flag in cell_merge[source.start_col: source.end_col])):
            continue
        target = previous_owners.get(source.start_col)
        if target is None or (target.start_col, target.end_col) != (
            source.start_col,
            source.end_col,
        ):
            continue
        if id(source.cell) in transferred_sources or id(target.cell) in transferred_targets:
            continue
        for child in list(source.cell.children):
            target.cell.append(child.extract())
        transferred_sources.add(id(source.cell))
        transferred_targets.add(id(target.cell))
    if transferred_sources and (not _row_has_semantic_content(first_data_row)):
        _carry_rowspan_structure_to_next_row(rows2, header_count)
        first_data_row.extract()
        rows2.pop(header_count)
    return bool(transferred_sources)
