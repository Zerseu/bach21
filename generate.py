"""Compose a new piece with a trained model.

    python generate.py bach violin                     # 1000 notes, seed 0
    python generate.py bach violin --length 500 --seed 7

Writes note_output.txt, midi_output.mid and xml_output.musicxml to bach21data/<composer>/<instrument>/.
The same seed gives the same piece. Run train.py first.
"""
import argparse
import sys

from config import log
from model import DEFAULT_LENGTH, main_generate


def main():
    parser = argparse.ArgumentParser(description='Generate music with a trained bach21 model.')
    parser.add_argument('composer', help='composer the model was trained on, for example bach')
    parser.add_argument('instruments', nargs='*', help='instrument(s) the model was trained on, for example violin')
    parser.add_argument('--length', type=int, default=DEFAULT_LENGTH,
                        help=f'number of notes (and rests) to generate (default: {DEFAULT_LENGTH})')
    parser.add_argument('--seed', type=int, default=0, help='random seed for the prompt and sampling (default: 0)')
    parser.add_argument('--kind', default='note', help='config.ini section / model kind (default: note)')
    parser.add_argument('--keep-prompt', action='store_true',
                        help='also write the real corpus notes used to prime the model, before the new ones')
    arguments = parser.parse_args()
    if arguments.length < 1:
        parser.error('--length must be at least 1')
    try:
        paths = main_generate(arguments.composer, arguments.instruments, kind=arguments.kind,
                              length=arguments.length, seed=arguments.seed, keep_prompt=arguments.keep_prompt)
    except FileNotFoundError as error:
        sys.exit(str(error))
    log('Wrote:\n  ' + '\n  '.join(paths))


if __name__ == '__main__':
    main()
