from stream.file_stream import FileStream


class FileStreamImplicit(FileStream):
    """
    Description
        A class which creates a data
        stream with implicit ratings out of a file dataset.
    """
    def __init__(self, path, sep=" "):
        """
        Description
            FileStreamImplicit's constructor.

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
            A function which returns a parsed implicit
            rating.

        Arguments
            :param rating_arr: A tuple of the form (user, item).
            :type rating_arr: tuple
        """
        user_id = int(rating_arr[0])
        item_id = int(rating_arr[1])
        return (user_id, item_id)
