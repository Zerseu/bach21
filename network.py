"""The neural network: token embedding, one LSTM layer, optional causal self-attention, output layer.

Shapes, with B the batch size, T the window length and H = hidden_size:

    token ids (B, T) -> Embedding (B, T, H) -> [LoRA identity projection] -> Dropout
      -> LSTM (B, T, H) -> CausalSelfAttention + residual + LayerNorm (B, T, H)
      -> Dropout -> Linear (B, T, vocabulary) logits

With tie_weights the Linear layer reuses the embedding matrix, so the plain model has
(H + 1) * vocabulary + 8 * H * (H + 1) + 4 * H * (H + 1) + 2 * H weights (embedding and output bias,
LSTM, attention, LayerNorm): 193 * vocabulary + 445,056 for H = 192.

Three options make the first and last layer aware of what a token means musically. They read the pitch
and the duration out of every token, so they need note tokens (the ``note`` model kind).

structured_embedding
    A plain embedding gives each of the ~800 tokens an unrelated vector: the network must learn on its
    own that ``C5_0.25`` and ``C5_0.5`` are the same pitch, and learns little about rare tokens. Here a
    token's vector is a sum of three parts:

        vector(token) = token part + pitch part(its pitch) + duration part(its duration)

    Tokens share the part for their pitch and the part for their duration; the token part only holds
    what is special about the combination. With tie_weights the output layer uses the same sum.
    Extra weights: (pitches + durations) * H, about 16k for Bach violin.

pitch_relations
    Tells the network how pitches relate, which it would otherwise have to rediscover in every key:
    - a pitch-class part in the token vector (all C's share it, whatever the octave);
    - an input embedding of the interval, in semitones, from the previous note;
    - an output score per interval: the score for "k semitones from the last note" is added to every
      token whose pitch lies at that interval, so a melodic habit learned in one key holds in all.
    Extra weights: (13 + 2 * MAX_INTERVAL + 2) * H + (2 * MAX_INTERVAL + 1) * (H + 1), about 22k.

pattern_pointers (experimental, off in config.ini)
    Lets the network point back at the last POINTER_SPAN notes. For k = 1..POINTER_SPAN it scores
    - "the pitch from k notes ago again": added to all tokens with that pitch;
    - "the interval from k notes ago again": added to all tokens whose pitch continues the melody by the
      interval played k notes ago (k = 1 repeats the last step);
    - "the duration from k tokens ago again": added to all tokens with that duration.
    Extra weights: 3 * POINTER_SPAN * (H + 1), about 7k. It lowers the held-out loss a little (0.03
    nats), but the generated music did not get better (see docs/results.md).
"""
from typing import Optional

import torch
import torch.nn.functional as F
from torch.nn import Module

# Training runs on the GPU when PyTorch sees one; generation always moves the model to the CPU.
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Intervals are clipped to +/- this many semitones (two octaves); larger leaps share the outermost value.
MAX_INTERVAL = 24
# Pitch class used for rests in the pitch-class embedding (0..11 are C..B).
REST_PITCH_CLASS = 12
# pattern_pointers can point this many notes (for pitches and intervals) or tokens (for durations) back.
POINTER_SPAN = 12


class CausalSelfAttention(Module):
    """Multi-head self-attention where each position only sees itself and earlier positions,
    followed by a residual connection and LayerNorm."""

    def __init__(self, d_model: int, n_heads: int = 4):
        super().__init__()
        self.mha = torch.nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln = torch.nn.LayerNorm(d_model)
        # Cached mask, not saved with the model (persistent=False), so model files stay compatible.
        self.register_buffer('_mask', None, persistent=False)

    def _causal_mask(self, length: int, device) -> torch.Tensor:
        """(length, length) bool mask, True above the diagonal: position i may not attend to j > i."""
        if self._mask is None or self._mask.size(0) != length:
            self._mask = torch.triu(torch.ones(length, length, device=device), diagonal=1).bool()
        return self._mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, d_model) -> (B, T, d_model)."""
        mask = self._causal_mask(x.size(1), x.device)
        y, _ = self.mha(x, x, x, attn_mask=mask)
        return self.ln(x + y)


class TorchModule(Module):
    """Next-token model. ``forward`` maps token ids (batch, time) to logits (batch, time, vocabulary);
    the logits at position t predict token t + 1.

    The attribute names below are the keys of saved model files, so renaming them breaks loading
    existing ``*_model.torch`` files.

    map_direct / map_reverse are the token <-> id maps; they are kept on the module for convenience but
    are not part of the saved state (the vocabulary is saved separately as ``<kind>_vocabulary.json``).
    The LoRA arguments only matter when lora_enable is True; see ``apply_lora``.

    token_features is ``(midi pitches, durations)`` with one entry per token id, as
    ``sequences.note_token_features`` returns it: the MIDI pitch number of the token (-1 for a rest) and
    its duration in grid steps. It is required by structured_embedding, pitch_relations and
    pattern_pointers (see the module docstring) and ignored otherwise. Everything derived from it is a
    non-persistent buffer: it is rebuilt from the vocabulary when a model is loaded and is not stored in
    the model file.
    """

    def __init__(self, vocabulary_size: int, map_direct: dict[str, int], map_reverse: dict[int, str], hidden_size: int,
                 lora_enable: bool = False, lora_r: int = 8, lora_alpha: int = 16, lora_dropout: float = 0.05,
                 lora_targets: str = 'ih,hh,out,emb', add_causal_attn: bool = True, attn_heads: int = 4,
                 dropout: float = 0.25, output_dropout: float = 0.0, tie_weights: bool = False,
                 token_features: Optional[tuple[list[int], list[int]]] = None,
                 structured_embedding: bool = False, pitch_relations: bool = False,
                 pattern_pointers: bool = False):
        super().__init__()
        if (structured_embedding or pitch_relations or pattern_pointers) and token_features is None:
            raise ValueError('structured_embedding, pitch_relations and pattern_pointers need '
                             'token_features (note tokens).')
        self.vocabulary_size = vocabulary_size
        self.map_direct = map_direct
        self.map_reverse = map_reverse
        self.hidden_size = hidden_size
        self.tie_weights = tie_weights
        self.structured_embedding = structured_embedding
        self.pitch_relations = pitch_relations
        self.pattern_pointers = pattern_pointers

        # The creation order matters: it decides how the seeded random generator initialises the weights.
        # The layers of the music-aware options are created last, so a model without them starts
        # from exactly the same weights as before they existed.
        self.embedding = torch.nn.Embedding(num_embeddings=vocabulary_size, embedding_dim=hidden_size)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.lstm = torch.nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)
        self.output_dropout = torch.nn.Dropout(p=output_dropout)
        self.linear = torch.nn.Linear(in_features=hidden_size, out_features=vocabulary_size)
        if tie_weights:
            # Input and output embeddings share one matrix: fewer weights, and rare tokens learn from both ends.
            torch.nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
            self.linear.weight = self.embedding.weight

        self.emb_post = None   # only created by apply_lora with the 'emb' target
        self.post_attn = CausalSelfAttention(hidden_size, n_heads=attn_heads) if add_causal_attn else None

        self.emb_pitch = self.emb_duration = self.emb_pitch_class = self.emb_interval = self.rel_out = None
        self.ptr_pitch = self.ptr_interval = self.ptr_duration = None
        if structured_embedding or pitch_relations or pattern_pointers:
            self._init_music_layers(token_features)

        if lora_enable:
            self.apply_lora(lora_r, lora_alpha, lora_dropout, lora_targets)

        self.to(DEVICE)

    def _init_music_layers(self, token_features: tuple[list[int], list[int]]):
        """Creates the lookup tables and the extra layers of structured_embedding, pitch_relations and
        pattern_pointers."""
        midi = torch.tensor(token_features[0], dtype=torch.long)       # (vocabulary,), -1 for rests
        steps = torch.tensor(token_features[1], dtype=torch.long)      # (vocabulary,), duration in grid steps
        if len(midi) != self.vocabulary_size or len(steps) != self.vocabulary_size:
            raise ValueError('token_features must have one entry per vocabulary token.')
        # The distinct pitches (the rest, -1, sorts first) and durations of the vocabulary, and for every
        # token the row of its pitch / duration in those lists.
        pitches, token_pitch = torch.unique(midi, sorted=True, return_inverse=True)
        durations, token_duration = torch.unique(steps, sorted=True, return_inverse=True)
        self.register_buffer('token_midi', midi, persistent=False)
        self.register_buffer('token_pitch', token_pitch, persistent=False)
        self.register_buffer('token_duration', token_duration, persistent=False)
        self.register_buffer('token_pitch_class',
                             torch.where(midi >= 0, midi % 12, torch.full_like(midi, REST_PITCH_CLASS)),
                             persistent=False)
        self.register_buffer('pitch_midi', pitches, persistent=False)   # (pitches,), MIDI number per pitch row
        # The inverse of pitch_midi: MIDI number -> pitch row, -1 for pitches the vocabulary does not have.
        midi_row = torch.full((128,), -1, dtype=torch.long)
        for row, number in enumerate(pitches.tolist()):
            if number >= 0:
                midi_row[number] = row
        self.register_buffer('midi_row', midi_row, persistent=False)
        # Which tokens have which pitch and which duration, as a 0/1 matrix with one row per pitch followed
        # by one row per duration: (pitches + durations, vocabulary). Scores that the network gives to a
        # pitch or to a duration are handed to all tokens that have it by one matrix product (see
        # _spread_over_tokens), which is several times faster than indexing the scores token by token.
        membership = torch.zeros(len(pitches) + len(durations), self.vocabulary_size)
        tokens = torch.arange(self.vocabulary_size)
        membership[token_pitch, tokens] = 1.0
        membership[len(pitches) + token_duration, tokens] = 1.0
        self.register_buffer('token_membership', membership, persistent=False)
        self.pitch_count = len(pitches)
        self.duration_count = len(durations)

        # A token vector is the sum of `parts` tables. Each is initialised 1 / sqrt(parts) as wide as the
        # single table of the plain model, so the sum starts with the same spread and the LSTM sees
        # inputs of the usual size.
        parts = 1 + (2 if self.structured_embedding else 0) + (1 if self.pitch_relations else 0)
        scale = parts ** -0.5

        def table(rows: int) -> torch.nn.Embedding:
            layer = torch.nn.Embedding(rows, self.hidden_size)
            if self.tie_weights:
                torch.nn.init.uniform_(layer.weight, -0.1 * scale, 0.1 * scale)
            else:
                torch.nn.init.normal_(layer.weight, 0.0, scale)
            return layer

        with torch.no_grad():
            self.embedding.weight.mul_(scale)
        if self.structured_embedding:
            self.emb_pitch = table(len(pitches))
            self.emb_duration = table(len(durations))
        if self.pitch_relations:
            self.emb_pitch_class = table(13)
            # Row 0 means "no interval" (the token is a rest, or no note came before it); rows 1.. are the
            # intervals -MAX_INTERVAL..+MAX_INTERVAL.
            self.emb_interval = torch.nn.Embedding(2 * MAX_INTERVAL + 2, self.hidden_size)
            torch.nn.init.uniform_(self.emb_interval.weight, -0.1, 0.1)
            # One score per interval -MAX_INTERVAL..+MAX_INTERVAL from the last note to the next one.
            self.rel_out = torch.nn.Linear(self.hidden_size, 2 * MAX_INTERVAL + 1)
        if self.pattern_pointers:
            # One score per look-back distance 1..POINTER_SPAN. They start at zero, so an untrained model
            # ignores the pointers and they only grow where repeating really helps.
            self.ptr_pitch = torch.nn.Linear(self.hidden_size, POINTER_SPAN)
            self.ptr_interval = torch.nn.Linear(self.hidden_size, POINTER_SPAN)
            self.ptr_duration = torch.nn.Linear(self.hidden_size, POINTER_SPAN)
            for layer in (self.ptr_pitch, self.ptr_interval, self.ptr_duration):
                torch.nn.init.zeros_(layer.weight)
                torch.nn.init.zeros_(layer.bias)

    def apply_lora(self, lora_r: int = 8, lora_alpha: int = 16, lora_dropout: float = 0.05,
                   lora_targets: str = 'ih,hh,out,emb'):
        """Wraps the current weights with LoRA adapters. Call it after loading a trained base model when
        fine-tuning, since the wrapped base weights are frozen.

        lora_targets is a comma-separated subset of: ih, hh (LSTM input and recurrent weights),
        out (output layer), emb (adds an adapted identity projection after the embedding).
        """
        from lora import inject_lora_into_linear, inject_lora_into_lstm
        targets = tuple(x.strip() for x in lora_targets.split(',') if x.strip())
        if 'emb' in targets:
            self.emb_post = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=False)
            torch.nn.init.eye_(self.emb_post.weight)
            inject_lora_into_linear(self.emb_post, r=lora_r, alpha=lora_alpha, dropout=lora_dropout)
        if 'ih' in targets or 'hh' in targets:
            inject_lora_into_lstm(self.lstm, r=lora_r, alpha=lora_alpha, dropout=lora_dropout,
                                  targets=tuple(x for x in ('ih', 'hh') if x in targets))
        if 'out' in targets:
            inject_lora_into_linear(self.linear, r=lora_r, alpha=lora_alpha, dropout=lora_dropout)
        self.to(DEVICE)

    def _shared_token_parts(self) -> Optional[torch.Tensor]:
        """(vocabulary, H): the parts of every token's vector that it shares with other tokens (its pitch,
        duration and pitch class), or None when neither music-aware option is on."""
        shared = None
        if self.structured_embedding:
            shared = self.emb_pitch.weight[self.token_pitch] + self.emb_duration.weight[self.token_duration]
        if self.pitch_relations:
            pitch_class = self.emb_pitch_class.weight[self.token_pitch_class]
            shared = pitch_class if shared is None else shared + pitch_class
        return shared

    def _last_pitches(self, x: torch.LongTensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """For token ids (B, T): each token's MIDI pitch (-1 for a rest), the pitch of the last note at or
        before each position, and whether there is such a note (False while a window has only had rests).

        The last note is found without a Python loop: every note position keeps its index, every rest
        gets -1, and a running maximum then holds the index of the most recent note.
        """
        midi = self.token_midi[x]
        positions = torch.arange(x.size(1), device=x.device).expand_as(x)
        last_position = torch.cummax(torch.where(midi >= 0, positions, torch.full_like(positions, -1)), dim=1).values
        has_last = last_position >= 0
        last = midi.gather(1, last_position.clamp(min=0))
        return midi, last, has_last

    def _recent_notes(self, midi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The pitches of the last POINTER_SPAN + 1 notes at or before each position.

        midi is (B, T) with -1 for rests. Returns `recent` (B, T, POINTER_SPAN + 1), where recent[..., k]
        is the MIDI pitch of the note k notes back (k = 0 is the last note itself, rests are skipped),
        and `valid` of the same shape, False where the window has not had that many notes yet.

        Without a Python loop: number the notes of each window 1, 2, 3, ... (a running count), store each
        note's pitch at its number, and look up the numbers count, count - 1, ..., count - POINTER_SPAN.
        """
        batch, length = midi.shape
        is_note = midi >= 0
        count = is_note.long().cumsum(dim=1)                         # notes up to and including position t
        by_number = torch.zeros(batch, length + 1, dtype=torch.long, device=midi.device)
        # Rests write to slot 0, which no valid look-up reads (numbers start at 1).
        by_number.scatter_(1, torch.where(is_note, count, torch.zeros_like(count)), midi.clamp(min=0))
        numbers = count[:, :, None] - torch.arange(POINTER_SPAN + 1, device=midi.device)[None, None, :]
        valid = numbers >= 1
        recent = by_number.gather(1, numbers.clamp(min=0).reshape(batch, -1)).reshape(batch, length, -1)
        return recent, valid

    def _pointer_scores(self, x: torch.LongTensor, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The pattern_pointers scores: (per pitch (B, T, pitches), per duration (B, T, durations)).

        h is the network's output state (B, T, H). See the module docstring for the three kinds of pointer.
        """
        batch, length = x.shape
        recent, valid = self._recent_notes(self.token_midi[x])
        per_pitch = torch.zeros(batch, length, self.pitch_count, dtype=h.dtype, device=h.device)

        # "The pitch from k notes ago again", k = 1..POINTER_SPAN: the score goes to that pitch's row.
        # Recent pitches come from input tokens, so their rows always exist.
        rows = self.midi_row[recent[..., 1:]].clamp(min=0)
        per_pitch = per_pitch.scatter_add(2, rows, self.ptr_pitch(h) * valid[..., 1:])

        # "The interval from k notes ago again": that interval led from note k to note k - 1 (counting
        # back), so the pitch it points at is last + recent[k - 1] - recent[k]. It may lie outside the
        # vocabulary's pitches (no row), in which case the score is dropped.
        target = recent[..., :1] + recent[..., :-1] - recent[..., 1:]
        rows = self.midi_row[target.clamp(0, 127)]
        usable = valid[..., 1:] & (target >= 0) & (target <= 127) & (rows >= 0)
        per_pitch = per_pitch.scatter_add(2, rows.clamp(min=0), self.ptr_interval(h) * usable)

        # "The duration from k tokens ago again". For the token after position t, k tokens ago is
        # position t - k + 1, so the history at t is [t, t - 1, ..., t - POINTER_SPAN + 1] (rests count).
        duration = self.token_duration[x]
        history = F.pad(duration, (POINTER_SPAN - 1, 0)).unfold(1, POINTER_SPAN, 1).flip(-1)   # (B, T, span)
        exists = (torch.arange(length, device=x.device)[:, None]
                  >= torch.arange(POINTER_SPAN, device=x.device)[None, :])[None, :, :]            # t - k + 1 >= 0
        per_duration = torch.zeros(batch, length, self.duration_count, dtype=h.dtype, device=h.device)
        per_duration = per_duration.scatter_add(2, history, self.ptr_duration(h) * exists)
        return per_pitch, per_duration

    def _spread_over_tokens(self, per_pitch: Optional[torch.Tensor],
                            per_duration: Optional[torch.Tensor]) -> torch.Tensor:
        """Turns scores per pitch (B, T, pitches) and / or per duration (B, T, durations) into scores per
        token (B, T, vocabulary): every token gets the score of its pitch plus the score of its duration.
        """
        if per_duration is None:
            return per_pitch @ self.token_membership[:self.pitch_count]
        if per_pitch is None:
            return per_duration @ self.token_membership[self.pitch_count:]
        return torch.cat([per_pitch, per_duration], dim=-1) @ self.token_membership

    def forward(self, x: torch.LongTensor) -> torch.FloatTensor:
        """Token ids (B, T) -> logits (B, T, vocabulary). Dropout is only active in train() mode."""
        shared = self._shared_token_parts()
        if shared is None:
            e = self.embedding(x)
        else:
            e = F.embedding(x, self.embedding.weight + shared)
        if self.pitch_relations:
            midi, last, has_last = self._last_pitches(x)
            # Interval from the note before position t to the note at t. `last` shifted right by one is
            # the last note heard strictly before t.
            previous = F.pad(last[:, :-1], (1, 0), value=0)
            has_previous = F.pad(has_last[:, :-1], (1, 0), value=False)
            interval = (midi - previous).clamp(-MAX_INTERVAL, MAX_INTERVAL) + MAX_INTERVAL + 1
            interval = torch.where((midi >= 0) & has_previous, interval, torch.zeros_like(interval))
            e = e + self.emb_interval(interval)
        if self.emb_post is not None:
            e = self.emb_post(e)
        e = self.dropout(e)
        h, _ = self.lstm(e)   # every window starts from a zero state; nothing is carried between windows
        if self.post_attn is not None:
            h = self.post_attn(h)
        h = self.output_dropout(h)

        if shared is None or not self.tie_weights:
            # Without tied weights the output layer keeps its own plain matrix; the shared parts then
            # only act on the input side.
            logits = self.linear(h)
        else:
            # self.linear.weight is the embedding matrix (wrapped by LoRA if the 'out' target is used).
            logits = F.linear(h, self.linear.weight + shared, self.linear.bias)
        # The music-aware output scores are collected per pitch and per duration first and handed to the
        # tokens in one step at the end.
        per_pitch = per_duration = None
        if self.pitch_relations:
            # Scores per interval, moved to the pitches they point at from the last note: pitch row p gets
            # the score of interval (pitch_midi[p] - last). Rests and windows without a note so far get 0.
            relative = self.rel_out(h)                                                    # (B, T, 2R + 1)
            index = (self.pitch_midi[None, None, :] - last[:, :, None] + MAX_INTERVAL).clamp(0, 2 * MAX_INTERVAL)
            per_pitch = relative.gather(2, index)                                         # (B, T, pitches)
            per_pitch = per_pitch * ((self.pitch_midi >= 0)[None, None, :] & has_last[:, :, None])
        if self.pattern_pointers:
            pointer_pitch, per_duration = self._pointer_scores(x, h)
            per_pitch = pointer_pitch if per_pitch is None else per_pitch + pointer_pitch
        if per_pitch is not None or per_duration is not None:
            logits = logits + self._spread_over_tokens(per_pitch, per_duration)
        return logits
