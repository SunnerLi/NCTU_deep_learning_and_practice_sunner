import argparse

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--caption_model', type=str, default="show_tell", 
        help='show_tell, show_attend_tell, adaatt, topdown')
    parser.add_argument('--num_layers', type=int, default=1, help='number of layers in the RNN')
    parser.add_argument('--batch_size', type=int, default=16, help='minibatch size')
    parser.add_argument('--epoch', type=int, default=1, help='number of epochs')
    parser.add_argument('--beam_size', type=int, default=1, help='The size of beam search') 

    # Parse and validate
    args = parser.parse_args()
    assert args.num_layers > 0, "num_layers should be greater than 0"
    assert args.batch_size > 0, "batch_size should be greater than 0"
    assert args.beam_size > 0, "beam_size should be greater than 0"

    return args