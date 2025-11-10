from model_train import imblance

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                        help='input batch size for training [default: 100]')
    parser.add_argument('--epochs', type=int, default=[500, 200], metavar='N',
                        help='number of epochs to train [default: 500]')
    parser.add_argument('--lr', type=float, default=[0.008, 0.003], metavar='LR',
                        help='learning rate')
    parser.add_argument('--R', type=int, default=5, metavar='N',
                        help='the number of KNN neighbors')
    parser.add_argument('--annealing_step', type=int, default=50, metavar='N',
                        help='gradually increase the value of lambda from 0 to 1')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='choose CUDA device [default: cuda:0]')

    args = parser.parse_args()

    dataset_name = 'PIE.mat'
    before_acc, after_acc = imblance(dataset_name=dataset_name, ratio=0.8, args=args)
    print('before: ====> acc: {:.4f}'.format(before_acc))
    print('after: ====> acc: {:.4f}'.format(after_acc))
