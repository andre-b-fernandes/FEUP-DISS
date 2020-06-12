
from stream.file_stream import FileStream


class FileStreamExplicit(FileStream):
    """
    Description
        A class which creates a data
        stream with explicit ratings out of a file dataset.
    """
    def __init__(self, path, sep=" "):
        """
        Description
            FileStreamExplicit's constructor.

        Arguments
            :param path: The path to the file.
            :type path: string
            :param sep: A character which separates the field in each line.
            :type sep: string
        """
        super().__init__(path, sep)

    def _parse_rating(self, rating_arr):
        """
        Description
            A function which returns a parsed explicit
            rating.

        Arguments
            :param rating_arr: A tuple of the form (user, item).
            :type rating_arr: tuple
        """
        user_id = int(rating_arr[0])
        item_id = int(rating_arr[1])
        value = float(rating_arr[2])
        return (user_id, item_id, value)
