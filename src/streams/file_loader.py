# Assuming files as (user_id, item_id, rating)
import time


class FileStream:

    def __init__(self, path, sep=" "):
        self.stream = self._parse_file(path, sep)

    def _parse_file(self, path, sep):
        stream = []
        with open(path, "r") as f:
            line = f.readline()
            while line:
                stream_arr = line.split(sep)
                user_id = int(stream_arr[0])
                item_id = int(stream_arr[1])
                rating = float(stream_arr[2])
                stream.append((user_id, item_id, rating))
                line = f.readline()
            f.close()
        return stream

    def process_stream(self, model):
        for rating in self.stream:
            print("New rating entering: " + str(rating))
            start_time = time.time()
            model.new_rating(rating)
            end_time = time.time()
            elapsed = end_time - start_time
            print("Elapsed time on rating: " + str(elapsed) + " seconds.")
        return model

    def process_stream_eval_anim(self, eval_class, model_class, anim_class):
        start_time = time.time()
        model = model_class()
        evaluator = eval_class(model)
        end_time = time.time()
        elapsed = end_time - start_time
        print("Empty model generated... in " + str(elapsed) + " seconds.")
        animation = anim_class(self.stream, evaluator)
        animation.show()
