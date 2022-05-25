
def printDictionnary(dict):
    """
    Print elements of a a dictionnary

    :param (dictionnary) dict: the dictionnary we want to display

    :return: (None)
    """
    for key, value in dict.items() :
        print(key + " : " + str(value))