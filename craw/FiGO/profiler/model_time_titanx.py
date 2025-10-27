from profiler.model_time import model_time_dict


def setup_static_model_time():
    model_time_dict['efficientdet-d0'] = 1 / 36.20
    model_time_dict['efficientdet-d1'] = 1 / 29.69
    model_time_dict['efficientdet-d2'] = 1 / 26.50
    model_time_dict['efficientdet-d3'] = 1 / 22.73
    model_time_dict['efficientdet-d4'] = 1 / 14.75
    model_time_dict['efficientdet-d5'] = 1 / 7.11
    model_time_dict['efficientdet-d6'] = 1 / 5.30
    model_time_dict['efficientdet-d7'] = 1 / 3.73