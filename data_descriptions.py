def load_data_description(filename):
    """
    A function that opens a .txt file stored in Data_descriptions folder and read it's content
    :param filename: name of file in Data_descriptions folder
    :type filename: str
    :return: content of .txt file
    :rtype: str
    """
    with open("Data_descriptions/" + str(filename) + ".txt", "r") as f:
        data_description = f.read()
    return data_description