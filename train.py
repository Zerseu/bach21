"""Train the note model for one composer and instrument.

    python train.py bach violin
    python train.py bach violin --retrain

The model is saved as bach21data/<composer>/<instrument>/note_model.torch, with the per-epoch losses in
note_log.csv. Settings come from the [note] section of config.ini.
"""
import argparse

from model import main_train


def main():
    parser = argparse.ArgumentParser(description='Train a bach21 model.')
    parser.add_argument('composer', help='composer name as used in the corpus cache, for example bach')
    parser.add_argument('instruments', nargs='*',
                        help='instrument(s) to train on, for example violin (default: every part)')
    parser.add_argument('--kind', default='note', help='config.ini section / model kind (default: note)')
    parser.add_argument('--retrain', action='store_true', help='train again even if a model already exists')
    arguments = parser.parse_args()
    main_train(arguments.composer, arguments.instruments, kind=arguments.kind, retrain=arguments.retrain)


if __name__ == '__main__':
    main()
