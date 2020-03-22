# Assuming files as (user_id, item_id, rating)
import time


def parse_file(path, sep):
    streams = []
    with open(path, "r") as f:
        dimension_users, dimension_items = 0, 0
        line = f.readline()
        while line:
            stream_arr = line.split(sep)
            user_id = int(stream_arr[0])
            item_id = int(stream_arr[1])
            rating = float(stream_arr[2])
            streams.append((user_id, item_id, rating))
            if user_id > dimension_users:
                dimension_users = user_id
            if item_id > dimension_items:
                dimension_items = item_id
            line = f.readline()
        f.close()

    return streams, dimension_users + 1, dimension_items + 1


def calc_streams_mat(path, sep):
    print("File: " + path)
    streams, dim_u, dim_i = parse_file(path, sep)
    matrix = [[None for _ in range(0, dim_i)] for _ in range(0, dim_u)]
    print("Empty matrix generated...")
    return streams, matrix


def process_file(path, model_class, sep=" "):
    streams, matrix = calc_streams_mat(path, sep)
    start_time = time.time()
    model = model_class(matrix)
    end_time = time.time()
    elapsed = end_time - start_time
    print("Empty model generated... in " + str(elapsed) + " seconds.")
    return process_streams(streams, model)


def process_file_evaluator(path, evaluator_class, model_class, sep=" "):
    streams, matrix = calc_streams_mat(path, sep)
    start_time = time.time()
    model = model_class(matrix)
    evaluator = evaluator_class(model)
    end_time = time.time()
    elapsed = end_time - start_time
    print("Empty model generated... in " + str(elapsed) + " seconds.")
    return process_streams(streams, evaluator)


def process_streams(streams, model):
    for stream in streams:
        print("New stream entering: " + str(stream))
        start_time = time.time()
        model.new_stream(stream)
        end_time = time.time()
        elapsed = end_time - start_time
        print("Elapsed time on stream: " + str(elapsed) + " seconds.")
    return model


def process_file_evaluator_graph(path, evaluator_class, model_class, animation_class, sep=" "):
    streams, matrix = calc_streams_mat(path, sep)
    start_time = time.time()
    model = model_class(matrix)
    evaluator = evaluator_class(model)
    end_time = time.time()
    elapsed = end_time - start_time
    print("Empty model generated... in " + str(elapsed) + " seconds.")
    animation = animation_class(streams, evaluator)
    animation.show()
