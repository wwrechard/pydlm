def get_time_series_from_dataframe(data, target_col, time_step_col=None):
    if time_step_col is None:
        return list(data.loc[:, target_col].values)

    data = data.sort_values(time_step_col)

    _time_series = data.loc[:, target_col].values
    _time_step = data.loc[:, time_step_col].values

    time_series = [_time_series[0]]

    for i in range(1, len(_time_step)):
        time_gap = _time_step[i] - _time_step[i - 1]
        time_series.extend([None] * (time_gap - 1))
        time_series.append(_time_series[i])

    return time_series
