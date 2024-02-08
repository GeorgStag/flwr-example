import flwr as fl
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix


def custom_fit_metrics_aggregation_fn(metrics):
    sum_score = 0
    sum_values = 0
    for i in metrics:
        lst = list(i)
        values = lst[0]
        score = lst[1]
        sum_score = sum_score + values*score['accuracy']
        sum_values = sum_values + values
    return {'accuracy': sum_score/sum_values}


def custom_evaluate_metrics_aggregation_fn(metrics):
    sum_score = 0
    sum_values = 0
    for i in metrics:
        lst = list(i)
        values = lst[0]
        score = lst[1]
        sum_score = sum_score + values*score['balanced_accuracy']
        sum_values = sum_values + values
    return {'balanced_accuracy': sum_score/sum_values}


if __name__ == "__main__":
    server_address = open("server\server_address", "r").read()
    strategy = fl.server.strategy.FedAvg(   fit_metrics_aggregation_fn = custom_fit_metrics_aggregation_fn, 
                                            evaluate_metrics_aggregation_fn = custom_evaluate_metrics_aggregation_fn
                                        )
    server = fl.server.Server(strategy=strategy, client_manager=fl.server.SimpleClientManager())
    fl.server.start_server(server_address=server_address, server=server)