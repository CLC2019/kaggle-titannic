import torch

class titanic_config():
    #CUDA = True
    device = torch.device("cpu")

    # logger
    print_step = 100
    test_step = 5
    #filename = datetime.now().__str__()[:-7]
    filename = '1'
    workdir = './history/{}'.format(filename)
    log = workdir + '/Log_{}.log'.format(filename)
    samples = workdir + '/samples'
    models = workdir + '/models'
    logger = None

    batch_size = 8
    epochs = 50
    save_step = 5
    learning_rate = 1e-3