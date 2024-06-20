def print_metrics(metrics):
    """
    Prints metrics in format for Latex.
    
    Parameters:
        metrics (dict): metrics to measure.
    """
    combined_dict = {key: [] for key in metrics[0]}
    for d in metrics:
        for key, value in d.items():
            combined_dict[key].append(round(value, 4))
    for metric, values in combined_dict.items():
        formatted_accuracies = " & ".join([f"{value:.4f}" for value in values])
        print(metric, formatted_accuracies)