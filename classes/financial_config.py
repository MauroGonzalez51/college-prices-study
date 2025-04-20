from dataclasses import dataclass
from typing import Tuple


@dataclass
class FinancialConfig(object):
    SEMESTER_WEEKS: int = 16
    MONTHLY_INCOME: int = 400_000
    SAVINGS: int = 100_000

    INCOME_VARIABILITY: float = 0.2
    EXTRA_INCOME_PROBABILITY: float = 0.01
    EXTRA_INCOME_RANGE: Tuple[int, int] = (
        int(MONTHLY_INCOME * 0.125), int(MONTHLY_INCOME * 0.25))

    WEEK_DAYS: int = 5
    LUNCH_PRICE: int = 16_500
    TRANSPORT_FARE: int = 3_400
    SNACK_PRICE_RANGE: Tuple[int, int] = (5_000, 15_000)

    LUNCHES_PER_WEEK: int = 3
    BUY_SNACK: bool = True
    SNACK_PROBABILITY: float = 0.5
    TRANSPORT_DAYS: int = WEEK_DAYS - 0

    EXTRA_EXPENSES_PROBABILITY: float = 0.4
    EXTRA_EXPENSES_RANGE: Tuple[int, int] = (50_000, 100_000)

    EMERGENCY_EXPENSES_PROBABILITY: float = 0.1
    EMERGENCY_RANGE: Tuple[int, int] = (100_000, 300_000)

    INFLATION_RATE: float = 0.0
