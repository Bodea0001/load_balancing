"""Распределение групп UE между Best-сотами с минимизацией"""

import argparse
import csv
import math
import random
from array import array
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, linprog, minimize


THRESHOLDS = {2600: -112.0, 2100: -112.0, 1800: -114.0, 900: -116.0, 800: -116.0}
RSRP_MIN = -120.0
RSRP_MAX = -105.0


@dataclass(frozen=True, slots=True)
class Cell:
    cell_id: str
    band: int
    bw: Decimal


@dataclass(frozen=True, slots=True)
class Subscriber:
    emr_id: str
    initial_cell: str
    initial_rsrp: float


@dataclass
class RadioData:
    """Смоделированные RSRP всех сот и единственная Best-сота каждого диапазона."""

    rsrp: dict[str, array]
    best: dict[int, list[str]]


def read_rows(path: Path, required: set[str]):
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: отсутствуют столбцы {', '.join(sorted(missing))}")
        for row in reader:
            if any(row[key] is None or not row[key].strip() for key in required):
                raise ValueError(f"{path}, строка {reader.line_num}: обязательное значение пусто")
            yield row


def read_cells(path: Path) -> list[Cell]:
    cells = []
    known = set()
    for row in read_rows(path, {"cell_id", "band", "bw"}):
        cell = Cell(row["cell_id"].strip(), int(row["band"]), Decimal(row["bw"]))
        if not cell.cell_id or cell.cell_id in known:
            raise ValueError(f"Пустой или повторяющийся cell_id: {cell.cell_id!r}")
        if cell.band not in THRESHOLDS:
            raise ValueError(f"Неизвестный диапазон {cell.band}; доступны {list(THRESHOLDS)}")
        if not cell.bw.is_finite() or cell.bw <= 0:
            raise ValueError(f"{cell.cell_id}: bw должна быть положительной")
        known.add(cell.cell_id)
        cells.append(cell)
    if not cells:
        raise ValueError("Файл сот пуст")
    return cells


def read_subscribers(path: Path, cells: list[Cell]) -> list[Subscriber]:
    by_id = {cell.cell_id: cell for cell in cells}
    users = []
    for number, row in enumerate(read_rows(path, {"initial_cell", "initial_rsrp"}), start=1):
        cell_id = row["initial_cell"].strip()
        if cell_id not in by_id:
            raise ValueError(f"Неизвестная исходная сота: {cell_id}")
        rsrp = float(row["initial_rsrp"])
        if not math.isfinite(rsrp):
            raise ValueError("Нужен конечный initial_rsrp")
        users.append(Subscriber(f"emr_{number}", by_id[cell_id].cell_id, rsrp))
    return users


def parse_shares(text: str | None) -> dict[str, float]:
    """Прочитать доли вида ключ:вес,ключ:вес; нормировка выполняется при распределении."""
    shares = {}
    for item in text.split(",") if text else []:
        key, value = item.rsplit(":", 1)
        key, weight = key.strip(), float(value)
        if not key or key in shares or not math.isfinite(weight) or weight < 0:
            raise ValueError(f"Некорректная или повторяющаяся доля: {item}")
        shares[key] = weight
    return shares


def split_counts(total: int, weights: dict) -> dict:
    """Получить целые количества с заданными долями и точной суммой total."""
    weight_sum = sum(weights.values())
    if weight_sum <= 0 or not math.isfinite(weight_sum):
        raise ValueError("Сумма весов должна быть положительной и конечной")
    expected = {key: total * weight / weight_sum for key, weight in weights.items()}
    counts = {key: math.floor(value) for key, value in expected.items()}
    order = sorted(weights, key=lambda key: expected[key] - counts[key], reverse=True)
    for key in order[:total - sum(counts.values())]:
        counts[key] += 1
    return counts


def parse_coverage(text: str | None, cells: list[Cell]) -> dict[int, float]:
    """Прочитать проценты абонентов с Best RSRP не ниже порога диапазона."""
    coverage = {int(key): value for key, value in parse_shares(text).items()}
    bands = {cell.band for cell in cells}
    if any(band not in bands or not 0 <= value <= 100 for band, value in coverage.items()):
        raise ValueError("Coverage: нужны имеющиеся диапазоны и проценты от 0 до 100")
    return coverage


def draw_rsrp(rng: random.Random, band: int, covered: bool | None = None) -> float:
    """Получить случайный RSRP в общем диапазоне либо по нужную сторону порога."""
    if covered is None:
        return rng.uniform(RSRP_MIN, RSRP_MAX)
    if covered:
        return rng.uniform(THRESHOLDS[band], RSRP_MAX)
    return rng.uniform(math.nextafter(RSRP_MIN, math.inf),
                       math.nextafter(THRESHOLDS[band], -math.inf))


def simulate_radio(cells: list[Cell], user_count: int, seed: int) -> RadioData:
    """Сгенерировать случайные RSRP с равными долями Best-сот в каждом диапазоне."""
    rng = random.Random(f"rsrp:{seed}")
    rsrp = {cell.cell_id: array("d") for cell in cells}
    best = {}
    for band in sorted({cell.band for cell in cells}, reverse=True):
        band_cells = [cell for cell in cells if cell.band == band]
        weights = {cell.cell_id: 1 for cell in band_cells}
        counts = split_counts(user_count, weights)
        winners = [cell_id for cell_id, count in counts.items() for _ in range(count)]
        rng.shuffle(winners)
        best[band] = winners
        for row, winner in enumerate(winners):
            values = [draw_rsrp(rng, band) for _ in band_cells]
            maximum = max(range(len(values)), key=lambda i: values[i])
            winner_index = next(i for i, cell in enumerate(band_cells) if cell.cell_id == winner)
            values[winner_index], values[maximum] = values[maximum], values[winner_index]
            for cell, value in zip(band_cells, values):
                rsrp[cell.cell_id].append(value)
    return RadioData(rsrp, best)


@dataclass(frozen=True)
class Group:
    """Количество UE с одним и тем же набором доступных Best-сот."""

    cell_ids: tuple[str, ...]
    ue_count: float


def parse_group_counts(text: str, cells: list[Cell]) -> list[Group]:
    """Прочитать количества UE в группах вида сота+сота:количество,сота:количество."""
    by_id = {cell.cell_id: cell for cell in cells}
    cell_order = {cell.cell_id: index for index, cell in enumerate(cells)}
    groups = []
    known_groups = set()
    for names, count in parse_shares(text).items():
        cell_ids = tuple(cell_id.strip() for cell_id in names.split("+"))
        for cell_id in cell_ids:
            if cell_id not in by_id:
                raise ValueError(f"Неизвестная сота в группе: {cell_id}")
        if len({by_id[cell_id].band for cell_id in cell_ids}) != len(cell_ids):
            raise ValueError(f"Группа {names}: допустима только одна Best-сота каждого диапазона")
        cell_ids = tuple(sorted(cell_ids, key=lambda cell_id: cell_order[cell_id]))
        if cell_ids in known_groups:
            raise ValueError(f"Группа задана повторно: {names}")
        known_groups.add(cell_ids)
        if count > 0:
            groups.append(Group(cell_ids, count))
    return groups


@dataclass
class BalanceResult:
    """Итоговые количества UE, коэффициенты распределения и значение функции."""

    cell_counts: dict[str, float]
    coefficients: dict[tuple[str, ...], dict[str, float]]
    target_load: float
    objective: float


def make_groups(cells: list[Cell], radio: RadioData) -> tuple[list[Group], dict[str, float]]:
    """Сформировать группы и отдельные назначения UE, не прошедших пороги."""
    bands = sorted(radio.best, reverse=True)
    user_count = len(radio.best[cells[0].band])
    counts = Counter()
    below_threshold_counts = Counter()
    for row in range(user_count):
        available = []
        for band in bands:
            cell_id = radio.best[band][row]
            if radio.rsrp[cell_id][row] >= THRESHOLDS[band]:
                available.append(cell_id)
        if not available:
            best_cell = max(
                (radio.best[band][row] for band in bands),
                key=lambda cell_id: radio.rsrp[cell_id][row],
            )
            below_threshold_counts[best_cell] += 1
        else:
            counts[tuple(available)] += 1
    groups = [Group(cell_ids, count) for cell_ids, count in counts.items()]
    return groups, dict(below_threshold_counts)


def balance_groups(
    cells: list[Cell], groups: list[Group], preassigned_ue: dict[str, float],
) -> BalanceResult:
    """Найти доли k с минимумом суммы BW * (C - C_target)**2"""
    # Группы уже сформированы по доступным Best-сотам. Здесь RSRP не пересчитывается.
    # Каждый UE учтён один раз: либо в одной из groups, либо в preassigned_ue.

    # Порядок элементов во всех массивах совпадает с порядком сот в cells.
    # cell_index переводит идентификатор соты в номер строки матрицы.
    cell_index = {cell.cell_id: index for index, cell in enumerate(cells)}
    bandwidths = np.array([float(cell.bw) for cell in cells])
    total_ue = sum(group.ue_count for group in groups) + sum(preassigned_ue.values())

    # Общая цель — одинаковое число UE на 1 МГц. Поэтому сота с полосой 40 МГц
    # должна получить вдвое больше UE, чем сота с полосой 20 МГц.
    # Отклонение от этих количеств допускается и штрафуется целевой функцией.
    target_load = total_ue / bandwidths.sum()
    target_counts = bandwidths * target_load
    coefficients = {}

    # UE ниже порогов уже назначены в свои лучшие соты. Их количество входит
    # в нагрузку, но само назначение при оптимизации остаётся фиксированным.
    fixed_counts = np.array([preassigned_ue.get(cell.cell_id, 0) for cell in cells], dtype=float)
    shared_groups = []

    # У группы с одной доступной сотой доля этой соты всегда равна 1.
    # В решатель передаются только группы, которые можно делить между сотами.
    for group in groups:
        if not np.isfinite(group.ue_count) or group.ue_count <= 0:
            raise ValueError("Количество UE в группе должно быть положительным и конечным")
        if not group.cell_ids:
            raise ValueError("Для каждой группы UE должна быть задана хотя бы одна сота")
        if len(set(group.cell_ids)) != len(group.cell_ids):
            raise ValueError("Сота не должна повторяться внутри группы")
        for cell_id in group.cell_ids:
            if cell_id not in cell_index:
                raise ValueError(f"Неизвестная сота в группе: {cell_id}")
        if len(group.cell_ids) == 1:
            cell_id = group.cell_ids[0]
            fixed_counts[cell_index[cell_id]] += group.ue_count
            coefficients[group.cell_ids] = {cell_id: 1.0}
        else:
            shared_groups.append(group)

    cell_counts = fixed_counts.copy()
    if shared_groups:
        # Один неизвестный коэффициент k задаёт долю одной группы в одной соте.
        # Например, для 20 UE с доступными сотами A и B нужны k_A и k_B:
        # k_A + k_B = 1, а количества UE будут равны 20 * k_A и 20 * k_B.
        variable_count = sum(len(group.cell_ids) for group in shared_groups)
        cell_rows = []
        group_rows = []
        ue_counts = []
        initial_coefficients = np.zeros(variable_count)
        group_slices = []
        start = 0

        # Каждый столбец соответствует коэффициенту k одной группы и одной соты.
        for row, group in enumerate(shared_groups):
            stop = start + len(group.cell_ids)
            # Срез сохраняет позиции коэффициентов группы в общем векторе k,
            # чтобы после поиска вернуть найденные доли к соответствующей группе.
            group_slices.append(slice(start, stop))
            # Для начала поиска SciPy нужно стартовое распределение. Деление поровну
            # сразу даёт допустимые доли: каждая >= 0, сумма по группе равна 1.
            initial_coefficients[start:stop] = 1 / len(group.cell_ids)
            for cell_id in group.cell_ids:
                cell_rows.append(cell_index[cell_id])
                group_rows.append(row)
                ue_counts.append(group.ue_count)
            start = stop

        columns = np.arange(variable_count)
        # count_matrix переводит доли в количества UE и суммирует их по сотам.
        # В строке соты A у столбца k_A будет размер группы, например 20.
        # Поэтому count_matrix @ k даёт количество распределённых UE каждой соты.
        count_matrix = sparse.coo_matrix(
            (ue_counts, (cell_rows, columns)), shape=(len(cells), variable_count),
        ).tocsc()
        # group_matrix суммирует доли внутри каждой группы. В её строке стоят
        # единицы у коэффициентов этой группы. Условие group_matrix @ k = 1
        # означает, что все UE группы распределены ровно один раз.
        group_matrix = sparse.coo_matrix(
            (np.ones(variable_count), (group_rows, columns)),
            shape=(len(shared_groups), variable_count),
        ).tocsc()

        # Матрицы хранят только существующие связи «группа — сота».
        # Нули для отсутствующих связей не занимают место в разреженном формате.

        # Разница между уже закреплённым и целевым количеством UE каждой соты.
        count_deviation = fixed_counts - target_counts
        # Общий масштаб весов сохраняет минимум и сокращает время решения на проверенных задачах.
        weights = bandwidths / bandwidths.max()
        group_totals = np.ones(len(shared_groups))

        # Сначала ищется любое распределение, точно попадающее в целевые количества UE.
        # Квадраты отклонений тогда равны нулю: улучшать такой результат уже некуда.
        # linprog получает нулевую линейную цель и проверяет выполнение ограничений:
        #   group_matrix @ k = 1 — все UE каждой группы распределены;
        #   count_matrix @ k = -count_deviation — вместе с фиксированными UE
        #   получается целевое количество в каждой соте.
        # vstack соединяет строки этих двух систем, concatenate — их правые части.
        solution = linprog(
            np.zeros(variable_count),
            A_eq=sparse.vstack([group_matrix, count_matrix], format="csc"),
            b_eq=np.concatenate([group_totals, -count_deviation]),
            bounds=(0.0, 1.0),
            method="highs",
            options={"primal_feasibility_tolerance": 1e-9,
                     "dual_feasibility_tolerance": 1e-9},
        )
        if solution.success:
            k = solution.x
        else:
            # Если попадание в цели не найдено, подбираются наименьшие отклонения.
            # Первые variable_count переменных — доли k, остальные — отклонения d (deviation)
            # каждой соты от цели. Здесь d выражается в UE без нормировки.
            # Отдельные d нужны, чтобы функция штрафа зависела только от одного числа
            # на соту. Так её производные остаются простыми даже при тысячах долей k.
            # Связи между долями и отклонениями целиком переносятся в ограничения.

            # Верхний блок матрицы задаёт group_matrix @ k = 1.
            # Нижний блок задаёт -count_matrix @ k + d = fixed_counts - target_counts.
            # После переноса слагаемых: d = fixed_counts + count_matrix @ k - target_counts.
            # То есть d — разница между итоговым и целевым количеством UE.
            # Единичная матрица связывает каждую строку соты с её собственной d.
            constraint_matrix = sparse.bmat(
                [[group_matrix, None], [-count_matrix, sparse.eye(len(cells))]],
                format="csc",
            )
            rhs = np.concatenate([group_totals, count_deviation])
            # Начальные d вычисляются из начальных k, чтобы оба набора переменных
            # сразу согласовывались между собой и выполняли линейные ограничения.
            initial_deviation = count_deviation + count_matrix @ initial_coefficients
            initial_values = np.concatenate([initial_coefficients, initial_deviation])

            # hessian — матрица вторых производных функции штрафа: по k — 0, по d — 2*weight.
            # Она постоянна и диагональна; вычисляется один раз для выбора шага решателем.
            hessian = sparse.diags(
                np.concatenate([np.zeros(variable_count), 2 * weights]), format="csc",
            )

            def objective(values):
                """Вычислить сумму взвешенных квадратов отклонений количества UE."""
                # По этому числу решатель сравнивает варианты распределения:
                # меньший штраф означает более близкое попадание в целевые количества UE.
                # Функция: sum(weight * d**2)
                deviation = values[variable_count:]
                return np.dot(weights * deviation, deviation)

            def gradient(values):
                """Вычислить производные по коэффициентам и отклонениям сот."""
                # gradient — вектор первых производных: по k — 0, по d — 2*weight*d.
                # Передаётся решателю для выбора направления уменьшения штрафа.
                return np.concatenate([np.zeros(variable_count),
                                       2 * weights * values[variable_count:]])

            # trust-constr получает точные производные и разреженные ограничения.
            # Поскольку функция квадратичная, hessian подходит на любой итерации.
            # gtol задаёт точность условий оптимальности и выполнения ограничений;
            # xtol — минимальный размер области поиска следующего шага.
            # barrier_* управляют внутренним барьером для ограничения k >= 0.
            # maxiter ограничивает число итераций, а не время работы.
            solution = minimize(
                objective,
                initial_values,
                jac=gradient,
                # SciPy запрашивает hessian через функцию от текущих переменных.
                # Здесь для любой точки подходит одна заранее вычисленная матрица.
                hess=lambda values: hessian,
                method="trust-constr",
                # Доли k неотрицательны. Вместе с суммой долей группы, равной 1,
                # это даёт k <= 1. Отклонения d могут иметь любой знак:
                # положительные означают превышение цели, отрицательные — недобор.
                bounds=Bounds(
                    np.concatenate([np.zeros(variable_count), np.full(len(cells), -np.inf)]),
                    np.full(variable_count + len(cells), np.inf),
                ),
                constraints=LinearConstraint(constraint_matrix, rhs, rhs),
                options={"gtol": 1e-10, "xtol": 1e-12, "barrier_tol": 1e-12,
                         "initial_barrier_parameter": 1e-10,
                         "initial_barrier_tolerance": 1e-10,
                         "maxiter": 1000, "sparse_jacobian": True},
            )
            if not solution.success:
                raise RuntimeError(f"Не удалось минимизировать функцию: {solution.message}")
            # В возвращаемых коэффициентах нужны только доли групп; вспомогательные d
            # отбрасываются. Итоговые количества UE ниже вычисляются из самих долей.
            k = solution.x[:variable_count]
        # Проверяются и сумма долей каждой группы, и границы отдельных долей.
        # Допуск учитывает погрешности вычислений с плавающей точкой.
        if (np.max(np.abs(group_matrix @ k - 1)) > 1e-8
                or np.min(k) < -1e-8 or np.max(k) > 1 + 1e-8):
            raise RuntimeError("Решение нарушает ограничения на доли групп")
        for group, columns in zip(shared_groups, group_slices):
            # После проверки убираются небольшие выходы за [0, 1], а сумма долей
            # приводится к 1, чтобы сохранялось точное количество UE каждой группы.
            k[columns] = np.clip(k[columns], 0, 1)
            k[columns] /= k[columns].sum()
            coefficients[group.cell_ids] = dict(zip(group.cell_ids, k[columns].tolist()))
        # Количества UE пересчитываются по окончательным долям после коррекции.
        cell_counts += count_matrix @ k

    objective_value = np.sum(bandwidths * (cell_counts - target_counts) ** 2)
    return BalanceResult(
        cell_counts=dict(zip(cell_index, cell_counts.tolist())),
        coefficients=coefficients,
        target_load=float(target_load),
        objective=float(objective_value),
    )


def print_report(
    cells: list[Cell], groups: list[Group], preassigned_ue: dict[str, float],
    result: BalanceResult,
) -> None:
    """Показать нагрузку сот и распределение абонентов каждой группы."""
    total_ue = sum(group.ue_count for group in groups) + sum(preassigned_ue.values())
    print(f"Всего UE: {total_ue:g}.")
    print(f"Target load: {result.target_load:.3f} UE/МГц.")
    print("\nСота     | BW, МГц | Цель, UE | Назначено, UE | Нагрузка, UE/МГц")
    for cell in cells:
        bw = float(cell.bw)
        count = result.cell_counts[cell.cell_id]
        print(f"{cell.cell_id:8} | {bw:7g} | {bw * result.target_load:8.2f} | "
              f"{count:13.2f} | {count / bw:.3f}")
    print("\nРаспределение абонентов по группам:")
    cell_order = {cell.cell_id: index for index, cell in enumerate(cells)}
    ordered_groups = sorted(
        groups,
        key=lambda group: (
            len(group.cell_ids),
            tuple(sorted(cell_order[cell_id] for cell_id in group.cell_ids)),
        ),
    )
    for group in ordered_groups:
        ordered_cells = sorted(group.cell_ids, key=lambda cell_id: cell_order[cell_id])
        group_cells = ", ".join(ordered_cells)
        distribution = "; ".join(
            f"{cell_id}: {group.ue_count * result.coefficients[group.cell_ids][cell_id]:.2f} UE"
            for cell_id in ordered_cells
        )
        print(f"Группа ({group_cells}), всего {group.ue_count:g} UE: "
              f"{distribution}.")
    if preassigned_ue:
        print("\nНазначены абоненты ниже порога:")
        for cell in cells:
            count = preassigned_ue.get(cell.cell_id, 0)
            if count:
                print(f"{cell.cell_id}: {count:g} UE.")
    print(f"\nЗначение функции: {result.objective:.3f}.")


def main() -> None:
    """Прочитать CSV, смоделировать Best-соты и распределить группы UE."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Группы задаются как сота+сота:количество,сота:количество. "
            "Неуказанные группы содержат 0 UE. Сумма --group-counts и --below-threshold "
            "должна совпадать с числом строк CSV абонентов. "
            "Без --group-counts состав групп определяется по случайным RSRP."
        ),
    )
    parser.add_argument("--cells", type=Path, default=Path("cells.csv"))
    parser.add_argument("--subscribers", type=Path, default=Path("emrs.csv"))
    parser.add_argument("--group-counts", help="Количество UE: 2600_1:10,2600_1+1800_1:20")
    parser.add_argument("--below-threshold", help="Назначения ниже всех порогов: 2600_1:5,1800_1:3")
    parser.add_argument("--seed", type=int, default=42, help="Фиксирует случайные RSRP")
    args = parser.parse_args()
    try:
        cells = read_cells(args.cells)
        users = read_subscribers(args.subscribers, cells)
        if args.group_counts is not None:
            groups = parse_group_counts(args.group_counts, cells)
            preassigned_ue = {cell_id: count for cell_id, count
                              in parse_shares(args.below_threshold).items() if count > 0}
            unknown = preassigned_ue.keys() - {cell.cell_id for cell in cells}
            if unknown:
                raise ValueError(f"Неизвестные соты в --below-threshold: {', '.join(sorted(unknown))}")
            total_ue = sum(group.ue_count for group in groups) + sum(preassigned_ue.values())
            if not math.isclose(total_ue, len(users), rel_tol=0, abs_tol=1e-8):
                raise ValueError(f"Задано {total_ue:g} UE, в файле абонентов — {len(users)}")
        else:
            if args.below_threshold is not None:
                raise ValueError("Параметр --below-threshold используется вместе с --group-counts")
            radio = simulate_radio(cells, len(users), args.seed)
            groups, preassigned_ue = make_groups(cells, radio)
        print(f"Сформировано групп доступных сот: {len(groups)}.")
        print("Подбираются доли распределения UE между сотами.")
        result = balance_groups(cells, groups, preassigned_ue)
        print_report(cells, groups, preassigned_ue, result)
    except (ValueError, ArithmeticError, RuntimeError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
