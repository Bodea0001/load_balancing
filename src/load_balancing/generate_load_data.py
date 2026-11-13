"""Генератор CSV с сотами и абонентами"""

import argparse
import csv
import random
from collections import Counter
from decimal import Decimal
from pathlib import Path

from load_balancing_scipy import Cell, THRESHOLDS, draw_rsrp, parse_coverage, parse_shares, split_counts


def make_cells(spec: str) -> list[Cell]:
    """Создать соты из описания частота:количество:полоса с идентификаторами частота_номер."""
    cells = []
    numbers = Counter()
    for item in spec.split(","):
        band_text, count_text, bw_text = item.split(":")
        band, count, bw = int(band_text), int(count_text), Decimal(bw_text)
        if band not in THRESHOLDS or count <= 0 or not bw.is_finite() or bw <= 0:
            raise ValueError(f"Некорректное описание сот: {item}")
        for _ in range(count):
            numbers[band] += 1
            cells.append(Cell(f"{band}_{numbers[band]}", band, bw))
    return cells


def initial_counts(total: int, cells: list[Cell], shares: dict[str, float]) -> dict[str, int]:
    """Распределить абонентов по долям сот либо по долям диапазонов."""
    ids = {cell.cell_id for cell in cells}
    bands = {str(cell.band) for cell in cells}
    if not shares:
        return split_counts(total, {cell.cell_id: 1 for cell in cells})
    if set(shares) <= ids:
        return split_counts(total, {cell.cell_id: shares.get(cell.cell_id, 0) for cell in cells})
    if set(shares) <= bands:
        by_band = split_counts(total, {band: shares.get(band, 0) for band in sorted(bands, key=int)})
        counts = {}
        for band, count in by_band.items():
            counts.update(split_counts(count, {cell.cell_id: 1 for cell in cells if str(cell.band) == band}))
        return counts
    raise ValueError("Initial shares: нужны известные cell_id либо частоты; смешивать их нельзя")


def generate(cells_path, users_path, cells, total, shares, coverage, seed) -> None:
    """Записать два CSV с точными целочисленными долями и воспроизводимыми значениями."""
    rng = random.Random(seed)
    counts = initial_counts(total, cells, shares)
    initial_cells = [cell_id for cell_id, count in counts.items() for _ in range(count)]
    rng.shuffle(initial_cells)
    by_id = {cell.cell_id: cell for cell in cells}
    flags = {}
    for band, percent in coverage.items():
        count = sum(counts[cell.cell_id] for cell in cells if cell.band == band)
        covered_counts = split_counts(count, {True: percent, False: 100 - percent})
        values = [flag for flag, size in covered_counts.items() for _ in range(size)]
        rng.shuffle(values)
        flags[band] = iter(values)
    with cells_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("cell_id", "band", "bw"))
        writer.writerows((cell.cell_id, cell.band, cell.bw) for cell in cells)
    with users_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("initial_cell", "initial_rsrp"))
        for cell_id in initial_cells:
            band = by_id[cell_id].band
            covered = next(flags[band]) if band in flags else None
            writer.writerow((cell_id, draw_rsrp(rng, band, covered)))
    print(f"Соты сохранены: {cells_path}; абоненты: {users_path}.")
    for cell in cells:
        print(f"{cell.cell_id}: {cell.band} МГц, {cell.bw:g} МГц полосы, "
              f"абонентов {counts[cell.cell_id]:,}.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Пример:
  python generate_load_data.py --users 1000 \\
      --bands "2600:3:40,2100:3:15,1800:3:20,900:3:5,800:3:5" \\
      --initial-shares "2600:10,800:90"

Bands: частота:количество_сот:полоса_каждой_соты. Идентификаторы: 2600_1, 2600_2, ...
Initial shares: либо диапазоны (2600:80,800:20), либо соты (2600_1:80,800_3:20).
Неуказанные значения получают нулевую долю. Без настройки все соты равноправны.
Доли округляются с сохранением общего числа абонентов; сумма весов произвольная.
Initial coverage управляет initial_rsrp внутри каждого исходного диапазона;
состав групп в расчёте задаётся отдельно через --group-counts.""",
    )
    parser.add_argument("--cells", type=Path, default=Path("cells.csv"))
    parser.add_argument("--subscribers", type=Path, default=Path("emrs.csv"))
    parser.add_argument("--users", type=int, default=1000)
    parser.add_argument("--bands", default="2600:3:40,2100:3:15,1800:3:20,900:3:5,800:3:5")
    parser.add_argument("--initial-shares", help="Исходные доли диапазонов или сот")
    parser.add_argument("--initial-coverage", help="Проценты initial_rsrp не ниже порога: 2600:20,800:100")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        if args.users < 0 or args.cells.resolve() == args.subscribers.resolve():
            raise ValueError("Число абонентов должно быть неотрицательным, пути файлов — разными")
        cells = make_cells(args.bands)
        generate(args.cells, args.subscribers, cells, args.users, parse_shares(args.initial_shares),
                 parse_coverage(args.initial_coverage, cells), args.seed)
    except (ValueError, ArithmeticError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
