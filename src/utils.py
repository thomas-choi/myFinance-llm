import numpy as np

def derive_label(next_high, next_low, last_high, last_low):
    if next_high > last_high and next_low > last_low:
        return 'UP'
    elif next_high < last_high and next_low < last_low:
        return 'DOWN'
    else:
        return 'SIDEWAY'

def forecast_to_label(forecast, last_day):
    # Assume forecast is dict with 'High', 'Low' (mean or sample)
    pred_high = forecast['High']  # Simplify; adjust per model
    pred_low = forecast['Low']
    last_high = last_day['High']
    last_low = last_day['Low']
    return derive_label(pred_high, pred_low, last_high, last_low)