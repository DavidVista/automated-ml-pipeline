import pandas as pd


def transform(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    X['TotalLateDays'] = X['NumberOfTime30-59DaysPastDueNotWorse'] + \
        X['NumberOfTime60-89DaysPastDueNotWorse'] + \
        X['NumberOfTimes90DaysLate']

    X['IncomeExpenseDifference'] = X['MonthlyIncome'] - X['MonthlyIncome'] * X['DebtRatio']
    X['90DaysLateLikelihood'] = (0.1 * X['NumberOfTime30-59DaysPastDueNotWorse'] +
                                 0.2 * X['NumberOfTime60-89DaysPastDueNotWorse'] +
                                 0.7 * X['NumberOfTimes90DaysLate'])

    X = X.drop(columns=['DebtRatio',
                        'MonthlyIncome',
                        'NumberOfTime30-59DaysPastDueNotWorse',
                        'NumberOfTime60-89DaysPastDueNotWorse',
                        'NumberOfTimes90DaysLate'])

    return X
