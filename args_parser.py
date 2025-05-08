import argparse


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-root', type=str, default='./data/')
    parser.add_argument('-result_path', type=str, default='./results/')
    parser.add_argument('-dataset', type=str, default='trento',
                        choices=['Berlin', 'Augsburg', 'MUUFL', 'trento'])
    parser.add_argument('--patch_size', type=int, default=11,
                        help='image patch size')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--test_ratio', type=int, default=0.2,
                        help='samples for validation')
    parser.add_argument('--num_class', type=int, default=7)

    args = parser.parse_args()
    return args