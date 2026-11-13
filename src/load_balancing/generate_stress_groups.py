"""Сформировать --group-counts для стрессового сценария с 15 сотами."""

import argparse
from itertools import product
from pathlib import Path
from random import Random

from load_balancing_scipy import Cell, read_cells, read_subscribers, split_counts


def make_stress_counts(
    cells: list[Cell], user_count: int, seed: int,
) -> dict[tuple[str, ...], int]:
    """Создать все пересечения с неравными размерами и перегруженными сотами."""
    by_band = {band: [cell.cell_id for cell in cells if cell.band == band]
               for band in sorted({cell.band for cell in cells}, reverse=True)}
    if len(by_band) != 5 or any(len(ids) != 3 for ids in by_band.values()):
        raise ValueError("Для сценария нужны 5 диапазонов по 3 соты")

    groups = [tuple(cell_id for cell_id in choice if cell_id)
              for choice in product(*[(None, *ids) for ids in by_band.values()])]
    counts = {group: 1 for group in groups if group}
    remaining = user_count - len(counts)
    if remaining <= 0:
        raise ValueError(f"В файле должно быть больше {len(counts)} абонентов")

    # В каждом диапазоне часть UE закрепляется только за первой сотой.
    for ids, percent in zip(by_band.values(), (26, 11, 13, 5, 5)):
        counts[(ids[0],)] += remaining * percent // 100

    # Крупная общая группа доступна только уже перегруженным сотам.
    heavy_group = tuple(ids[0] for ids in by_band.values())
    counts[heavy_group] += remaining * 30 // 100

    rng = Random(seed)
    weights = {group: 10 ** rng.uniform(-8, 0)
               for group in counts if len(group) > 1 and group != heavy_group}
    for group, count in split_counts(user_count - sum(counts.values()), weights).items():
        counts[group] += count

    target_load = user_count / sum(float(cell.bw) for cell in cells)
    by_id = {cell.cell_id: cell for cell in cells}
    for ids in by_band.values():
        cell = by_id[ids[0]]
        if counts[(cell.cell_id,)] <= float(cell.bw) * target_load:
            raise ValueError(
                "Для этих полос и количества UE не обеспечивается перегрузка всех диапазонов"
            )
    return counts


def main() -> None:
    """Вывести аргумент --group-counts, сохраняя общее число UE из CSV."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=Path, default=Path("cells.csv"))
    parser.add_argument("--subscribers", type=Path, default=Path("emrs.csv"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        cells = read_cells(args.cells)
        users = read_subscribers(args.subscribers, cells)
        counts = make_stress_counts(cells, len(users), args.seed)
        print(",".join(f"{'+'.join(group)}:{count}" for group, count in counts.items()))
    except (ValueError, ArithmeticError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
