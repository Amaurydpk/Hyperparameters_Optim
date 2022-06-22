
def printDictionnary(dict):
    """
    Print elements of a a dictionnary

    :param (dictionnary) dict: the dictionnary we want to display

    :return: (None)
    """
    optim = dict['optimizer']
    for key, value in dict.items():
        if key == 'listParamConvLayers':
            print(key + " : " + "\n\t(nOutputChannel, kernelSize, stride, padding, doPool)" + "\n\t" + str(value))
        
        elif key == 'optimParam1':
            if optim == 'Adam':
                print('beta1' + " : " + str(value))
            if optim == 'ASGD':
                print('lambda' + " : " + str(value))
            if optim == 'Adagrad':
                print('lr decay' + " : " + str(value))
            if optim == 'RMSprop':
                print('momentum' + " : " + str(value))
        
        elif key == 'optimParam2':
            if optim == 'Adam':
                print('beta2' + " : " + str(value))
            if optim == 'ASGD':
                print('alpha' + " : " + str(value))
            if optim == 'Adagrad':
                print('initial accumulator value' + " : " + str(value))
            if optim == 'RMSprop':
                print('alpha' + " : " + str(value))

        else:
            print(key + " : " + str(value))


def printHPset(dict):
    """
    Print elements of a a dictionnary

    :param (dictionnary) dict: the dictionnary we want to display

    :return: (None)
    """
    optim = dict['optimizer']
    for key, hp in dict.items():
        # if key == 'listParamConvLayers':
        #     print(key + " : " + "\n\t(nOutputChannel, kernelSize, stride, padding, doPool)" + "\n\t" + str(hp.value))
        
        # elif key == 'optimParam1':
        #     if optim == 'Adam':
        #         print('beta1' + " : " + str(hp.value))
        #     if optim == 'ASGD':
        #         print('lambda' + " : " + str(hp.value))
        #     if optim == 'Adagrad':
        #         print('lr decay' + " : " + str(hp.value))
        #     if optim == 'RMSprop':
        #         print('momentum' + " : " + str(hp.value))
        
        # elif key == 'optimParam2':
        #     if optim == 'Adam':
        #         print('beta2' + " : " + str(hp.value))
        #     if optim == 'ASGD':
        #         print('alpha' + " : " + str(hp.value))
        #     if optim == 'Adagrad':
        #         print('initial accumulator value' + " : " + str(hp.value))
        #     if optim == 'RMSprop':
        #         print('alpha' + " : " + str(hp.value))

        # else:
        print(hp)