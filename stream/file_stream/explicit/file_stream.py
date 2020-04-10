
from stream.file_stream import FileStream


class FileStreamExplicit(FileStream):
    def __init__(self, path, sep=" "):
        super().__init__(path, sep)

    def _parse_rating(self, rating_arr):
        user_id = int(rating_arr[0])
        item_id = int(rating_arr[1])
        value = float(rating_arr[2])
        return (user_id, item_id, value)
