"""Models package providing a unified factory for all sequence architectures."""

from .base import BaseSeq2Seq
from .transformer_seq2seq import TransformerSeq2Seq
from .lstm_seq2seq import LSTMSeq2Seq
from .cnn_seq2seq import CNNSeq2Seq
from .mamba_seq2seq import MambaSeq2Seq

def build_model(arch: str,
                vocab_src_size: int,
                vocab_tgt_size: int,
                pad_idx: int = 0,
                sos_idx: int = 1,
                eos_idx: int = 2,
                **kwargs) -> BaseSeq2Seq:
    """Builds and returns the requested model architecture."""
    arch = arch.lower()
    if arch in ['transformer', 'indicxlit']:
        return TransformerSeq2Seq(
            vocab_src_size=vocab_src_size,
            vocab_tgt_size=vocab_tgt_size,
            d_model=kwargs.get('d_model', 256),
            nhead=kwargs.get('nhead', 4),
            num_encoder_layers=kwargs.get('num_encoder_layers', 6),
            num_decoder_layers=kwargs.get('num_decoder_layers', 6),
            dim_feedforward=kwargs.get('dim_feedforward', 1024),
            dropout=kwargs.get('dropout', 0.1),
            pad_idx=pad_idx, sos_idx=sos_idx, eos_idx=eos_idx
        )
    elif arch in ['lstm', 'rnn', 'bilstm']:
        return LSTMSeq2Seq(
            vocab_src_size=vocab_src_size,
            vocab_tgt_size=vocab_tgt_size,
            embed_dim=kwargs.get('d_model', 256),
            hidden_dim=kwargs.get('hidden_dim', 256),
            num_layers=kwargs.get('num_layers', 3),
            dropout=kwargs.get('dropout', 0.1),
            pad_idx=pad_idx, sos_idx=sos_idx, eos_idx=eos_idx
        )
    elif arch in ['cnn', 'conv', 'convseq2seq']:
        return CNNSeq2Seq(
            vocab_src_size=vocab_src_size,
            vocab_tgt_size=vocab_tgt_size,
            embed_dim=kwargs.get('d_model', 256),
            num_layers=kwargs.get('num_layers', 6),
            kernel_size=kwargs.get('kernel_size', 3),
            dropout=kwargs.get('dropout', 0.1),
            pad_idx=pad_idx, sos_idx=sos_idx, eos_idx=eos_idx
        )
    elif arch in ['mamba', 'ssm']:
        return MambaSeq2Seq(
            vocab_src_size=vocab_src_size,
            vocab_tgt_size=vocab_tgt_size,
            d_model=kwargs.get('d_model', 256),
            d_state=kwargs.get('d_state', 16),
            num_layers=kwargs.get('num_layers', 4),
            dropout=kwargs.get('dropout', 0.1),
            pad_idx=pad_idx, sos_idx=sos_idx, eos_idx=eos_idx
        )
    else:
        raise ValueError(f"Unknown architecture '{arch}'. Supported: transformer, lstm, cnn, mamba")

__all__ = [
    'BaseSeq2Seq',
    'TransformerSeq2Seq',
    'LSTMSeq2Seq',
    'CNNSeq2Seq',
    'MambaSeq2Seq',
    'build_model'
]
