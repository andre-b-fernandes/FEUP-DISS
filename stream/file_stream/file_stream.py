# Assuming files as (user_id, item_id, rating)
class FileStream:
    """
    Description
        A class which encapsulates common logic for the
        implicit and explicit versions, and creates a data
        stream with ratings out of a file dataset.
    """
    def __init__(self, path, sep=" "):
        """
        Description
            FileStream's constructor

        Arguments
            :param path: The path to the file.
            :type path: string
            :param sep: A character which separates the field in each line.
            :type sep: string
        """
        self.stream = self._parse_file(path, sep)

    def _parse_file(self, path, sep):
        """
        Description
            A function which parsed a file and returns a
            data stream list.

        Arguments
            :param path: The path to the file.
            :type path: string
            :param sep: A character which separates the field in each line.
            :type sep: string
        """
        stream = []
        with open(path, "r") as f:
            line = f.readline()
            while line:
                rating_arr = line.split(sep)
                rating = self._parse_rating(rating_arr)
                stream.append((rating))
                line = f.readline()
            f.close()
        return stream

    def process_stream(self, model):
        """
        Description
            A function which processes a data stream
            with a recomendation model.

        Arguments
            :param path: A recommendation algorithm or evaluator.
            :type path: CollaborativeFiltering or PrequentialEvaluator.
        """
        it = 0
        for rating in self.stream:
            # print(f"New rating entering: {rating} -> Iter: {it}")
            it += 1
            model.new_rating(rating)
        return model
