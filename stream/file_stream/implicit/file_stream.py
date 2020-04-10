from stream.file_stream import FileStream


class FileStreamImplicit(FileStream):
    def __init__(self, path, sep=" "):
        super().__init__(path, sep)

    def _parse_rating(self, rating_arr):
        user_id = int(rating_arr[0])
        item_id = int(rating_arr[1])
        return (user_id, item_id)
